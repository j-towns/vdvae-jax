from functools import partial

import jax.numpy as jnp
from flax import linen as nn

from jax.util import safe_map
from jax import random
from jax import image
from jax import custom_jvp
from jax.scipy.special import logsumexp

import hps

map = safe_map


def gaussian_kl(mu1, mu2, logsigma1, logsigma2):
    return (-0.5 + logsigma2 - logsigma1
            + 0.5 * (jnp.exp(logsigma1) ** 2 + (mu1 - mu2) ** 2)
            / (jnp.exp(logsigma2) ** 2))

def gaussian_sample(mu, sigma, rng):
    return mu + sigma * random.normal(rng, mu.shape)

Conv1x1 = partial(nn.Conv, kernel_size=(1, 1), strides=(1, 1))
Conv3x3 = partial(nn.Conv, kernel_size=(3, 3), strides=(1, 1), padding='SAME')

def resize(img, shape):
    n, _, _, c = img.shape
    return image.resize(img, (n,) + shape + (c,), 'nearest')

# Logistic mixture distribution utils
def logistic_preprocess(nn_out):
    *batch, h, w, _ = nn_out.shape
    assert nn_out.shape[-1] % 10 == 0
    k = nn_out.shape[-1] // 10
    logit_weights, nn_out = jnp.split(nn_out, [k], -1)
    m, s, t = jnp.moveaxis(
        jnp.reshape(nn_out, tuple(batch) + (h, w, 3, k, 3)), (-2, -1), (-4, 0))
    assert m.shape == tuple(batch) + (k, h, w, 3)
    inv_scales = jnp.maximum(nn.softplus(s), 1e-7)
    return m, jnp.tanh(t), inv_scales, jnp.moveaxis(logit_weights, -1, -3)

def logistic_mix_logpmf(nn_out, img):
    m, t, inv_scales, logit_weights = logistic_preprocess(nn_out)
    img = jnp.expand_dims(img, -4)  # Add mixture dimension
    mean_red   = m[..., 0]
    mean_green = m[..., 1] + t[..., 0] * img[..., 0]
    mean_blue  = m[..., 2] + t[..., 1] * img[..., 0] + t[..., 2] * img[..., 1]
    means = jnp.stack((mean_red, mean_green, mean_blue), axis=-1)
    logprobs = jnp.sum(logistic_logpmf(img, means, inv_scales), -1)
    log_mix_coeffs = logit_weights - logsumexp(logit_weights, -3, keepdims=True)
    return jnp.sum(logsumexp(log_mix_coeffs + logprobs, -3), (-2, -1))

def logistic_logpmf(img, means, inv_scales):
    centered = img - means
    top    = -jnp.logaddexp(0,  (centered - 1 / 255) * inv_scales)
    bottom = -jnp.logaddexp(0, -(centered + 1 / 255) * inv_scales)
    mid = log1mexp(inv_scales / 127.5) + top + bottom
    return jnp.select([img == -1, img == 1], [bottom, top], mid)

def logistic_mix_sample(nn_out, rng):
    m, t, inv_scales, logit_weights = logistic_preprocess(nn_out)
    rng_mix, rng_logistic = random.split(rng)
    mix_idx = random.categorical(rng_mix, logit_weights, -3)
    def select_mix(arr):
        return jnp.squeeze(
            jnp.take_along_axis(
                arr, jnp.expand_dims(mix_idx, (-4, -1)), -4), -4)
    m, t, inv_scales = map(lambda x: jnp.moveaxis(select_mix(x), -1, 0),
                           (m, t, inv_scales))
    l = random.logistic(rng_logistic, m.shape) / inv_scales
    img_red   = m[0]                                     + l[0]
    img_green = m[1] + t[0] * img_red                    + l[1]
    img_blue  = m[2] + t[1] * img_red + t[2] * img_green + l[2]
    return jnp.stack([img_red, img_green, img_blue], -1)

@custom_jvp
def log1mexp(x):
    """Accurate computation of log(1 - exp(-x)) for x > 0."""
    # Method from
    # https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    return jnp.where(
        x > jnp.log(2), jnp.log1p(-jnp.exp(-x)), jnp.log(-jnp.expm1(-x)))

# Workaround for NaN gradients
log1mexp.defjvps(lambda t, _, x: t / jnp.expm1(x))

class DmolNet(nn.Module):
    H: hps.Hyperparams

    def setup(self):
        self.out_conv = Conv1x1(self.H.num_mixtures * 10,
                                precision=self.H.conv_precision)

    def loglik(self, px_z, x):
        return logistic_mix_logpmf(self.out_conv(px_z), x)

    def sample(self, px_z, rng):
        img = logistic_mix_sample(self.out_conv(px_z), rng)
        return jnp.round((jnp.clip(img, -1, 1) + 1) * 127.5).astype(jnp.uint8)
