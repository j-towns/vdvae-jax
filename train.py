import numpy as np
from functools import partial
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data import set_up_data
from vae import VAE
from train_helpers import (set_up_hyperparams, load_vaes, accumulate_stats,
                           save_model, linear_warmup, shard_batch,
                           clip_grad_norm)
from PIL import Image

from flax import jax_utils
from jax import tree_map, tree_multimap, device_get
from jax import grad, lax, pmap
from jax import random
import jax.numpy as jnp


def training_step(H, data_input, target, optimizer, ema, rng):
    def loss_fun(params):
        stats = VAE(H).apply({'params': params}, data_input, target, rng)
        return -stats['elbo'] * np.log(2), stats

    gradval, stats = lax.pmean(
        grad(loss_fun, has_aux=True)(optimizer.target), 'batch')
    gradval, grad_norm = clip_grad_norm(gradval, H.grad_clip)

    ll_nans = jnp.any(jnp.isnan(stats['log_likelihood']))
    kl_nans = jnp.any(jnp.isnan(stats['kl']))
    stats.update(log_likelihood_nans=ll_nans, kl_nans=kl_nans)

    learning_rate = H.lr * linear_warmup(H.warmup_iters)(optimizer.state.step)

    # only update if no rank has a nan and if the grad norm is below a specific
    # threshold
    def skip_update(_):
        # Only increment the step
        return optimizer.replace(
            state=optimizer.state.replace(step=optimizer.state.step + 1)), ema
    def update(_):
        optimizer_ = optimizer.apply_gradient(
            gradval, learning_rate=learning_rate)
        e_decay = H.ema_rate
        ema_ = tree_multimap(
            lambda e, p: e * e_decay + (1 - e_decay) * p, ema, optimizer.target)
        return optimizer_, ema_
    skip = (ll_nans | kl_nans | ((H.skip_threshold != -1)
                                 & ~(grad_norm < H.skip_threshold)))
    optimizer, ema = lax.cond(skip, skip_update, update, None)
    stats.update(skipped_updates=skip, grad_norm=grad_norm)
    return optimizer, ema, stats
# Would use donate_argnums=(3, 4) here but compilation never finishes
p_training_step = pmap(training_step, 'batch', static_broadcasted_argnums=0)

def eval_step(H, data_input, target, ema_params, rng):
    return lax.pmean(VAE(H).apply(
        {'params': ema_params}, data_input, target, rng), 'batch')
p_eval_step = pmap(eval_step, 'batch', static_broadcasted_argnums=0)

def synchronize(x):
    return lax.pmean(x, 'batch')
p_synchronize = pmap(synchronize, 'batch')

def get_sample_for_visualization(data, preprocess_fn, num, dataset):
    for x in DataLoader(data, batch_size=num):
        break
    orig_image = ((x[0] * 255.0).to(torch.uint8).permute(0, 2, 3, 1)
                  if dataset == 'ffhq_1024' else x[0])
    preprocessed = preprocess_fn(x)[0]
    return orig_image, preprocessed

def train_loop(
        H, data_train, data_valid, preprocess_fn, optimizer, ema,
        starting_epoch, logprint):
    rng = random.PRNGKey(H.seed_train)
    iterate = int(optimizer.state.step[0])
    train_sampler = DistributedSampler(
        data_train, num_replicas=H.host_count, rank=H.host_id)
    viz_batch_original, viz_batch_processed = get_sample_for_visualization(
        data_valid, preprocess_fn, H.num_images_visualize, H.dataset)
    early_evals = set([1] + [2 ** exp for exp in range(3, 14)])
    stats = []
    iters_since_starting = 0
    for epoch in range(starting_epoch, H.num_epochs):
        train_sampler.set_epoch(epoch)
        for x in DataLoader(
                data_train, batch_size=H.n_batch * H.device_count,
                drop_last=True, pin_memory=True, sampler=train_sampler):
            rng, iter_rng = random.split(rng)
            iter_rng = random.split(iter_rng, H.device_count)
            data_input, target = map(partial(shard_batch, H), preprocess_fn(x))
            t0 = time.time()
            optimizer, ema, training_stats = p_training_step(
                H, data_input, target, optimizer, ema, iter_rng)
            training_stats = device_get(
                tree_map(lambda x: x[0], training_stats))
            training_stats['iter_time'] = time.time() - t0
            stats.append(training_stats)
            if (iterate % H.iters_per_print == 0
                    or (iters_since_starting in early_evals)):
                logprint(model=H.desc, type='train_loss',
                         lr=H.lr * float(
                             linear_warmup(H.warmup_iters)(iterate)),
                         epoch=epoch, step=iterate,
                         **accumulate_stats(stats, H.iters_per_print))

            if (iterate % H.iters_per_images == 0
                    or (iters_since_starting in early_evals
                        and H.dataset != 'ffhq_1024')):
                write_images(H, ema, viz_batch_original,
                             viz_batch_processed,
                             f'{H.save_dir}/samples-{iterate}.png', logprint)

            iterate += 1
            iters_since_starting += 1
            if iterate % H.iters_per_save == 0:
                if np.isfinite(stats[-1]['elbo']):
                    logprint(model=H.desc, type='train_loss', epoch=epoch,
                             step=iterate,
                             **accumulate_stats(stats, H.iters_per_print))
                    fp = os.path.join(H.save_dir, 'latest')
                    logprint(f'Saving model@ {iterate} to {fp}')
                    save_model(fp, optimizer, ema, epoch, H)

            if iterate % H.iters_per_ckpt == 0:
                save_model(os.path.join(H.save_dir, f'iter-{iterate}'),
                           optimizer, ema, epoch, H)

        optimizer, ema = p_synchronize((optimizer, ema))
        if epoch % H.epochs_per_eval == 0:
            valid_stats = evaluate(H, ema, data_valid, preprocess_fn)
            logprint(model=H.desc, type='eval_loss', epoch=epoch, step=iterate,
                     **valid_stats)


def evaluate(H, ema_params, data_valid, preprocess_fn):
    rng = random.PRNGKey(H.seed_eval)
    stats_valid = []
    valid_sampler = DistributedSampler(
        data_valid, num_replicas=H.host_count, rank=H.host_id)
    for x in DataLoader(
            data_valid, batch_size=H.n_batch * H.device_count, drop_last=True,
            pin_memory=True, sampler=valid_sampler):
        rng, iter_rng = random.split(rng)
        iter_rng = random.split(iter_rng, H.device_count)
        data_input, target = map(partial(shard_batch, H), preprocess_fn(x))
        stats_valid.append(tree_map(lambda x: np.array(x[0]),
            p_eval_step(
                H, data_input, target, ema_params, iter_rng)))
    vals = [a['elbo'] for a in stats_valid]
    finites = np.array(vals)[np.isfinite(vals)]
    stats = dict(
        n_batches=len(vals), filtered_elbo=np.mean(finites),
        **{k: np.mean([a[k] for a in stats_valid]) for k in stats_valid[-1]})
    return stats


def write_images(H, ema_params, viz_batch_original, viz_batch_processed, fname,
                 logprint):
    rng = random.PRNGKey(H.seed_sample)
    ema_apply = partial(VAE(H).apply,
                        {'params': jax_utils.unreplicate(ema_params)})
    forward_get_latents = partial(ema_apply, method=VAE(H).forward_get_latents)
    forward_samples_set_latents = partial(
        ema_apply, method=VAE(H).forward_samples_set_latents)
    forward_uncond_samples = partial(
        ema_apply, method=VAE(H).forward_uncond_samples)

    zs = [s['z'] for s in forward_get_latents(viz_batch_processed, rng)]
    batches = [viz_batch_original.numpy()]
    mb = viz_batch_processed.shape[0]
    lv_points = np.floor(
        np.linspace(
            0, 1, H.num_variables_visualize + 2) * len(zs)).astype(int)[1:-1]
    for i in lv_points:
        batches.append(forward_samples_set_latents(mb, zs[:i], rng, t=0.1))
    for t in [1.0, 0.9, 0.8, 0.7][:H.num_temperatures_visualize]:
        batches.append(forward_uncond_samples(mb, rng, t=t))
    n_rows = len(batches)
    im = np.concatenate(batches, axis=0).reshape(
        (n_rows, mb, *viz_batch_processed.shape[1:])).transpose(
            [0, 2, 1, 3, 4]).reshape(
                [n_rows * viz_batch_processed.shape[1],
                 mb * viz_batch_processed.shape[2], 3])
    logprint(f'printing samples to {fname}')
    Image.fromarray(im).save(fname)


def run_test_eval(H, ema_params, data_test, preprocess_fn, logprint):
    print('evaluating')
    stats = evaluate(H, ema_params, data_test, preprocess_fn)
    print('test results')
    for k in stats:
        print(k, stats[k])
    logprint(type='test_loss', **stats)


def main():
    H, logprint = set_up_hyperparams()
    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
    optimizer, ema, epoch = load_vaes(H, logprint)
    if H.test_eval:
        run_test_eval(H, ema, data_valid_or_test, preprocess_fn, logprint)
    else:
        train_loop(H, data_train, data_valid_or_test, preprocess_fn, optimizer,
                   ema, epoch, logprint)


if __name__ == "__main__":
    main()
