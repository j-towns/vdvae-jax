from functools import partial
import itertools
from typing import Optional, Any

import numpy as np

import jax.numpy as jnp
from jax import random
from jax.util import safe_map

from flax import linen as nn

from vae_helpers import (Conv1x1, Conv3x3, gaussian_sample, DmolNet, resize,
                         gaussian_kl)
import hps


map = safe_map

# Want to be able to vary the scale of initialized parameters
def lecun_normal(scale):
    return nn.initializers.variance_scaling(
        scale, 'fan_in', 'truncated_normal')

class Block(nn.Module):
    middle_width: int
    out_width: int
    conv_precision: Any
    down_rate: int = 1
    residual: bool = False
    use_3x3: bool = True
    last_scale: bool = 1.

    @nn.compact
    def __call__(self, x):
        Conv1x1_ = partial(Conv1x1, precision=self.conv_precision)
        Conv3x3_ = partial(Conv3x3 if self.use_3x3 else Conv1x1,
                          precision=self.conv_precision)
        x_ = Conv1x1_(self.middle_width)(nn.gelu(x))
        x_ = Conv3x3_(self.middle_width)(nn.gelu(x_))
        x_ = Conv3x3_(self.middle_width)(nn.gelu(x_))
        x_ = Conv1x1_(
            self.out_width, kernel_init=lecun_normal(self.last_scale))(
                nn.gelu(x_))
        out = x + x_ if self.residual else x_
        if self.down_rate > 1:
            window_shape = 2 * (self.down_rate,)
            out = nn.avg_pool(out, window_shape, window_shape)
        return out

def parse_layer_string(s):
    layers = []
    for ss in s.split(','):
        if 'x' in ss:
            res, count = ss.split('x')
            layers.extend(int(count) * [(int(res), None)])
        elif 'm' in ss:
            res, mixin = ss.split('m')
            layers.append((int(res), int(mixin)))
        elif 'd' in ss:
            res, down_rate = ss.split('d')
            layers.append((int(res), int(down_rate)))
        else:
            res = int(ss)
            layers.append((res, 1))
    return layers

def pad_channels(t, new_width):
    return jnp.pad(t, (t.ndim - 1) * [(0, 0)] + [(0, new_width - t.shape[-1])])

def get_width_settings(s):
    mapping = {}
    if s:
        for ss in s.split(','):
            k, v = ss.split(':')
            mapping[k] = int(v)
    return mapping

class Encoder(nn.Module):
    H: hps.Hyperparams

    @nn.compact
    def __call__(self, x):
        H = self.H
        widths = get_width_settings(H.custom_width_str)
        x = Conv3x3(H.width, precision=H.conv_precision)(x)
        activations = {}
        activations[x.shape[1]] = x  # Spatial dimension
        for res, down_rate in parse_layer_string(H.enc_blocks):
            use_3x3 = res > 2  # Don't use 3x3s for 1x1, 2x2 patches
            width = widths.get(str(res), H.width)
            x = Block(int(width * H.bottleneck_multiple), width,
                      H.conv_precision, down_rate or 1, True, use_3x3)(x)
            new_res = x.shape[1]
            new_width = widths.get(str(new_res), H.width)
            x = x if (x.shape[3] == new_width) else pad_channels(x, new_width)
            activations[new_res] = x
        return activations

class DecBlock(nn.Module):
    H: hps.Hyperparams
    res: int
    mixin: Optional[int]
    n_blocks: int

    def setup(self):
        H = self.H
        width = self.width = get_width_settings(
            H.custom_width_str).get(str(self.res), H.width)
        use_3x3 = self.res > 2
        cond_width = int(width * H.bottleneck_multiple)
        self.zdim = H.zdim
        self.enc   = Block(cond_width, H.zdim * 2, H.conv_precision,
                           use_3x3=use_3x3)
        self.prior = Block(cond_width, H.zdim * 2 + width, H.conv_precision,
                           use_3x3=use_3x3, last_scale=0.)
        self.z_proj = Conv1x1(
            width, kernel_init=lecun_normal(np.sqrt(1 / self.n_blocks)),
            precision=H.conv_precision)
        self.resnet = Block(cond_width, width, H.conv_precision, residual=True,
                            use_3x3=use_3x3,
                            last_scale=np.sqrt(1 / self.n_blocks))
        self.z_fn = lambda x: self.z_proj(x)

    def sample(self, x, acts, rng):
        x = jnp.broadcast_to(x, acts.shape)
        qm, qv = jnp.split(self.enc(jnp.concatenate([x, acts], 3)), 2, 3)
        pm, pv, xpp = jnp.split(self.prior(x), [self.zdim, 2 * self.zdim], 3)
        z = gaussian_sample(qm, jnp.exp(qv), rng)
        kl = gaussian_kl(qm, pm, qv, pv)
        return z, x + xpp, kl

    def sample_uncond(self, x, rng, t=None, lvs=None):
        pm, pv, xpp = jnp.split(self.prior(x), [self.zdim, 2 * self.zdim], 3)
        return (gaussian_sample(pm, jnp.exp(pv) * (t or 1), rng)
                if lvs is None else lvs, x + xpp)

    def __call__(self, xs, activations, rng, get_latents=False):
        acts = activations[self.res]
        x = xs[self.res] if self.res in xs else 0
        if self.mixin is not None:
            # Assume width increases monotonically with depth
            x = x + resize(xs[self.mixin][..., :acts.shape[3]],
                           (self.res, self.res))
        z, x, kl = self.sample(x, acts, rng)
        return (self.resnet(x + self.z_fn(z)),
                (dict(kl=kl, z=z) if get_latents else dict(kl=kl)))

    def forward_uncond(self, xs, rng, t=None, lvs=None):
        assert self.res in xs or self.mixin is not None
        x = xs[self.res] if self.res in xs else 0
        if self.mixin is not None:
            # Assume width increases monotonically with depth
            x = x + resize(xs[self.mixin][..., :self.width],
                           (self.res, self.res))
        z, x = self.sample_uncond(x, rng, t, lvs)
        return self.resnet(x + self.z_fn(z))

class Decoder(nn.Module):
    H: hps.Hyperparams

    def setup(self):
        H = self.H
        resos = set()
        dec_blocks = []
        self.widths = get_width_settings(H.custom_width_str)
        blocks = parse_layer_string(H.dec_blocks)
        for res, mixin in blocks:
            dec_blocks.append(DecBlock(H, res, mixin, n_blocks=len(blocks)))
            resos.add(res)
        self.dec_blocks = dec_blocks
        self.bias_xs = [
            self.param(f'bias_xs_{i}', nn.initializers.zeros,
                       (res, res, self.widths.get(str(res), H.width)))
            for i, res in enumerate(sorted(resos)) if res <= H.no_bias_above]
        self.out_net = DmolNet(H)
        self.gain = self.param('gain', nn.initializers.ones, (H.width,))
        self.bias = self.param('bias', nn.initializers.zeros, (H.width,))
        self.final_fn = lambda x: x * self.gain + self.bias

    def get_bias_xs(self):
        return map(partial(getattr, self), self.bias_xs_names)

    def __call__(self, activations, rng, get_latents=False):
        stats = []
        xs = {a.shape[0]: a for a in self.bias_xs}
        for block in self.dec_blocks:
            rng, block_rng = random.split(rng)
            x, block_stats = block(xs, activations, block_rng, get_latents)
            stats.append(block_stats)
            xs[block.res] = x
        return self.final_fn(xs[self.H.image_size]), stats

    def forward_uncond(self, n, rng, t=None):
        xs = {a.shape[0]: jnp.broadcast_to(a, (n,) + a.shape)
              for a in self.bias_xs}
        for idx, block in enumerate(self.dec_blocks):
            t_block = t[idx] if isinstance(t, list) else t
            rng, block_rng = random.split(rng)
            xs[block.res] = block.forward_uncond(xs, block_rng, t_block)
        return self.final_fn(xs[self.H.image_size])

    def forward_manual_latents(self, n, latents, rng, t=None):
        xs = {a.shape[0]: jnp.broadcast_to(a, (n,) + a.shape)
              for a in self.bias_xs}
        for block, lvs in itertools.zip_longest(self.dec_blocks, latents):
            rng, block_rng = random.split(rng)
            xs[block.res] = block.forward_uncond(xs, block_rng, t, lvs)
        return self.final_fn(xs[self.H.image_size])

class VAE(nn.Module):
    H: hps.Hyperparams

    def setup(self):
        self.encoder = Encoder(self.H)
        self.decoder = Decoder(self.H)

    def __call__(self, x, x_target, rng):
        px_z, stats = self.decoder(self.encoder(x), rng)
        ndims = np.prod(x.shape[1:])
        ln2 = np.log(2)
        ll = self.decoder.out_net.loglik(px_z, x_target).mean() / (ndims * ln2)
        kl = sum(s['kl'].sum((1, 2, 3)).mean() for s in stats) / (ndims * ln2)
        return dict(elbo=ll - kl, log_likelihood=ll, kl=kl)

    def forward_get_latents(self, x, rng):
        _, stats = self.decoder(self.encoder(x), rng, get_latents=True)
        return stats

    def forward_uncond_samples(self, size, rng, t=None):
        latent_rng, obs_rng = random.split(rng)
        px_z = self.decoder.forward_uncond(size, latent_rng, t=t)
        return self.decoder.out_net.sample(px_z, obs_rng)

    def forward_samples_set_latents(self, size, latents, rng, t=None):
        latent_rng, obs_rng = random.split(rng)
        px_z = self.decoder.forward_manual_latents(size, latents, latent_rng, t)
        return self.decoder.out_net.sample(px_z, obs_rng)
