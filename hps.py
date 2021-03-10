import dataclasses
import jax

HPARAMS_REGISTRY = {}


@dataclasses.dataclass(frozen=True)
class Hyperparams:
    adam_beta1: float = .9
    adam_beta2: float = .9
    axis_visualize: int = None
    bottleneck_multiple: float = .25
    conv_precision: str = dataclasses.field(
        default='default',
        metadata=dict(nargs='?', choices=['default', 'high', 'highest']))
    custom_width_str: str = ''
    data_root: str = './'
    dataset: str = 'cifar10'
    dec_blocks: str = None
    desc: str = 'test'
    device_count: int = jax.local_device_count()
    ema_rate: float = .999
    enc_blocks: str = None
    epochs_per_eval: int = 10
    grad_clip: float = 200.
    host_count: int = jax.host_count()
    host_id: int = jax.host_id()
    hps: str = None
    image_channels: int = None
    image_size: int = None
    iters_per_ckpt: int = 25000
    iters_per_images: int = 10000
    iters_per_print: int = 1000
    iters_per_save: int = 10000
    logdir: str = None
    log_wandb: bool = False
    lr: float = .00015
    n_batch: int = 32
    no_bias_above: int = 64
    num_depths_visualize: int = None
    num_epochs: int = 10000
    num_images_visualize: int = 8
    num_mixtures: int = 10
    num_temperatures_visualize: int = 3
    num_variables_visualize: int = 6
    restore_path: str = None
    save_dir: str = './saved_models'
    seed: int = 0
    seed_eval: int = None
    seed_init: int = None
    seed_sample: int = None
    seed_train: int = None
    skip_threshold: float = 400.
    test_eval: bool = False
    warmup_iters: float = 0.
    wd: float = 0.
    width: int = 512
    zdim: int = 16


cifar10 = dict(
    width=384,
    lr=0.0002,
    zdim=16,
    wd=0.01,
    dec_blocks="1x1,4m1,4x2,8m4,8x5,16m8,16x10,32m16,32x21",
    enc_blocks="32x11,32d2,16x6,16d2,8x6,8d2,4x3,4d4,1x3",
    warmup_iters=100,
    dataset='cifar10',
    n_batch=16,
    ema_rate=0.9999,
)
HPARAMS_REGISTRY['cifar10'] = cifar10

i32 = {
    **cifar10,
    **dict(
        dataset='imagenet32',
        ema_rate=0.999,
        dec_blocks="1x2,4m1,4x4,8m4,8x9,16m8,16x19,32m16,32x40",
        enc_blocks="32x15,32d2,16x9,16d2,8x8,8d2,4x6,4d4,1x6",
        width=512,
        n_batch=8,
        lr=0.00015,
        grad_clip=200.,
        skip_threshold=300.,
        epochs_per_eval=1
    )
}
HPARAMS_REGISTRY['imagenet32'] = i32

i64 = {
    **i32,
    **dict(
        n_batch=4,
        grad_clip=220.0,
        skip_threshold=380.0,
        dataset='imagenet64',
        dec_blocks="1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12",
        enc_blocks="64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5",
    )
}
HPARAMS_REGISTRY['imagenet64'] = i64

ffhq_256 = {
    **i64,
    **dict(
        n_batch=1,
        lr=0.00015,
        dataset='ffhq_256',
        epochs_per_eval=1,
        num_images_visualize=2,
        num_variables_visualize=3,
        num_temperatures_visualize=1,
        dec_blocks="1x2,4m1,4x3,8m4,8x4,16m8,16x9,32m16,32x21,64m32,64x13,128m64,128x7,256m128",
        enc_blocks="256x3,256d2,128x8,128d2,64x12,64d2,32x17,32d2,16x7,16d2,8x5,8d2,4x5,4d4,1x4",
        no_bias_above=64,
        grad_clip=130.,
        skip_threshold=180.,
    )
}
HPARAMS_REGISTRY['ffhq256'] = ffhq_256

ffhq1024 = {
    **ffhq_256,
    **dict(
        dataset='ffhq_1024',
        data_root='./ffhq_images1024x1024',
        epochs_per_eval=1,
        num_images_visualize=1,
        iters_per_images=25000,
        num_variables_visualize=0,
        num_temperatures_visualize=4,
        grad_clip=360.,
        skip_threshold=500.,
        num_mixtures=2,
        width=16,
        lr=0.00007,
        dec_blocks="1x2,4m1,4x3,8m4,8x4,16m8,16x9,32m16,32x20,64m32,64x14,128m64,128x7,256m128,256x2,512m256,1024m512",
        enc_blocks="1024x1,1024d2,512x3,512d2,256x5,256d2,128x7,128d2,64x10,64d2,32x14,32d2,16x7,16d2,8x5,8d2,4x5,4d4,1x4",
        custom_width_str="512:32,256:64,128:512,64:512,32:512,16:512,8:512,4:512,1:512",
    )
}
HPARAMS_REGISTRY['ffhq1024'] = ffhq1024


def parse_args_and_update_hparams(H, parser, s=None):
    H = dataclasses.replace(H, **vars(parser.parse_args(s)))
    hparam_sets = [x for x in H.hps.split(',') if x]
    for hp_set in hparam_sets:
        hps = HPARAMS_REGISTRY[hp_set]
        parser.set_defaults(**hps)
    return dataclasses.replace(H, **vars(parser.parse_args(s)))


def add_vae_arguments(parser):
    for f in dataclasses.fields(Hyperparams):
        kwargs = (dict(action='store_true') if f.type is bool and not f.default else
                  dict(default=f.default, type=f.type))
        parser.add_argument(f'--{f.name}', **kwargs, **f.metadata)

    return parser
