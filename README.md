# Very Deep VAEs in JAX/Flax
Implementation of the experiments in the paper [_Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images_](https://arxiv.org/abs/2011.10650) using [JAX](https://github.com/google/jax) and [Flax](https://github.com/google/flax), ported from the [official OpenAI PyTorch implementation](https://github.com/openai/vdvae).

I have tried to keep this implementation as close as possible to the original. I was able to re-use a large proportion of the code, including the data input pipeline, which still uses PyTorch. I recommend installing a CPU-only version of PyTorch for this.

Tested with JAX 0.2.10, Flax 0.3.0, PyTorch 1.7.1, NumPy 1.19.2.

From the paper, some model samples and a visualization of how it generates them:

![image](header-image.png)

# Setup
As well as JAX, Flax, NumPy and PyTorch, this implementation depends on [Pillow](https://pillow.readthedocs.io) and [scikit-learn](https://scikit-learn.org):
```
pip install pillow
pip install sklearn
```
Also, you'll have to download the data, depending on which one you want to run:
```
./setup_cifar10.sh
./setup_imagenet.sh imagenet32
./setup_imagenet.sh imagenet64
./setup_ffhq256.sh
./setup_ffhq1024.sh  /path/to/images1024x1024  # this one depends on you first downloading the subfolder `images_1024x1024` from https://github.com/NVlabs/ffhq-dataset on your own & running `pip install torchvision`
```

# Training models
Hyperparameters all reside in `hps.py`.
```bash
python train.py --hps cifar10
python train.py --hps imagenet32
python train.py --hps imagenet64
python train.py --hps ffhq256
python train.py --hps ffhq1024
```

# TODOs
- Implement support for 5 bit images which was used in the paper's FFHQ-256 experiments. 

# Known differences from the orignal
 - Instead of using the PyTorch default layer initializers we use
   the Flax defaults.
 - We haven't yet implemented support for the 'low_bit' hyperparameter.
 - Renamed rate/distortion to kl/loglikelihood.
 - In multihost configurations, checkpoints are saved to disk on all hosts.
 - Slight changes to DMOL loss.

# Things to be careful of
We tried to keep this implementation as close as possible to the author's [original Pytorch implementation](https://github.com/openai/vdvae). There are two potentially confusing things which we chose to preserve. Firstly, the `--n_batch` command line argument specifies the _per device_ batch size; on configurations with multiple GPUs/TPUs and multiple hosts this needs to be taken into account when comparing runs on different configurations. Secondly, some of the default hyperparameter settings in `hps.py` do not match the settings used for the paper's experiments, which are specified on page 15 of the paper.
