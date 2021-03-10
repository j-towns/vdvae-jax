import torch
import numpy as np
import dataclasses
import argparse
import os
import subprocess

from jax.interpreters.xla import DeviceArray

from hps import Hyperparams, parse_args_and_update_hparams, add_vae_arguments
from utils import logger
from data import mkdir_p
from vae import VAE

from jax import lax
import jax
from jax import random
import jax.numpy as jnp
from jax.util import safe_map

from flax import jax_utils
from flax.training import checkpoints
from flax.optim import Adam

map = safe_map


def save_model(path, optimizer, ema, epoch, H):
    optimizer, ema = jax_utils.unreplicate((optimizer, ema))
    checkpoints.save_checkpoint(path, (optimizer, epoch), optimizer.state.step)
    checkpoints.save_checkpoint(path + '_ema', ema, optimizer.state.step)
    from_log = os.path.join(H.save_dir, 'log.jsonl')
    to_log = f'{os.path.dirname(path)}/{os.path.basename(path)}-log.jsonl'
    subprocess.check_output(['cp', from_log, to_log])

def load_vaes(H, logprint):
    rng = random.PRNGKey(H.seed_init)
    init_rng, init_eval_rng = random.split(rng)
    init_batch = jnp.zeros((1, H.image_size, H.image_size, H.image_channels))
    ema = params = VAE(H).init({'params': init_rng}, init_batch, init_batch,
                               init_eval_rng)['params']
    optimizer = Adam(weight_decay=H.wd, beta1=H.adam_beta1,
                     beta2=H.adam_beta2).create(params)
    epoch = 0
    if H.restore_path:
        logprint(f'Restoring vae from {H.restore_path}')
        optimizer, epoch = checkpoints.restore_checkpoint(H.restore_path, (optimizer, epoch))
        ema = checkpoints.restore_checkpoint(H.restore_path + '_ema', ema)
    total_params = 0
    for p in jax.tree_flatten(optimizer.target)[0]:
        total_params += np.prod(p.shape)
    logprint(total_params=total_params, readable=f'{total_params:,}')
    optimizer, ema = jax_utils.replicate((optimizer, ema))
    return optimizer, ema, epoch

def accumulate_stats(stats, frequency):
    z = {}
    for k in stats[-1]:
        if k in ['log_likelihood_nans', 'kl_nans', 'skipped_updates']:
            z[k] = np.sum([a[k] for a in stats[-frequency:]])
        elif k == 'grad_norm':
            vals = [a[k] for a in stats[-frequency:]]
            finites = np.array(vals)[np.isfinite(vals)]
            if len(finites) == 0:
                z[k] = 0.0
            else:
                z[k] = np.max(finites)
        elif k == 'elbo':
            vals = [a[k] for a in stats[-frequency:]]
            finites = np.array(vals)[np.isfinite(vals)]
            z['elbo'] = np.mean(vals)
            z['elbo_filtered'] = np.mean(finites)
        elif k == 'iter_time':
            z[k] = (stats[-1][k] if len(stats) < frequency
                    else np.mean([a[k] for a in stats[-frequency:]]))
        else:
            z[k] = np.mean([a[k] for a in stats[-frequency:]])
    return z

def linear_warmup(warmup_iters):
    return lambda i: lax.min(1., i / warmup_iters)

def setup_save_dirs(H):
    save_dir = os.path.join(H.save_dir, H.desc)
    mkdir_p(save_dir)
    logdir = os.path.join(save_dir, 'log')
    return dataclasses.replace(
        H,
        save_dir=save_dir,
        logdir=logdir,
    )

def set_up_hyperparams(s=None):
    H = Hyperparams()
    parser = argparse.ArgumentParser()
    parser = add_vae_arguments(parser)
    H = parse_args_and_update_hparams(H, parser, s=s)
    H = setup_save_dirs(H)
    log = logger(H.logdir)
    if H.log_wandb:
        import wandb
        def logprint(*args, pprint=False, **kwargs):
            if len(args) > 0: log(*args)
            wandb.log({k: np.array(x) if type(x) is DeviceArray else x for k, x in kwargs.items()})
        wandb.init(config=dataclasses.asdict(H))
    else:
        logprint = log
        for i, k in enumerate(sorted(dataclasses.asdict(H))):
            logprint(type='hparam', key=k, value=getattr(H, k))
    np.random.seed(H.seed)
    torch.manual_seed(H.seed)
    logprint('training model', H.desc, 'on', H.dataset)
    H = dataclasses.replace(
        H,
        conv_precision = {'default': lax.Precision.DEFAULT,
                          'high': lax.Precision.HIGH,
                          'highest': lax.Precision.HIGHEST}[H.conv_precision],
        seed_init  =H.seed,
        seed_sample=H.seed + 1,
        seed_train =H.seed + 2 + H.host_id,
        seed_eval  =H.seed + 2 + H.host_count + H.host_id,
    )
    return H, logprint

def shard_batch(H, batch):
    return jnp.reshape(batch, (H.device_count, -1) + batch.shape[1:])

def clip_grad_norm(g, max_norm):
    # Simulates torch.nn.utils.clip_grad_norm_
    g, treedef = jax.tree_flatten(g)
    total_norm = jnp.linalg.norm(jnp.array(map(jnp.linalg.norm, g)))
    clip_coeff = jnp.minimum(max_norm / (total_norm + 1e-6), 1)
    g = [clip_coeff * g_ for g_ in g]
    return treedef.unflatten(g), total_norm
