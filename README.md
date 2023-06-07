# kcg-ml-sd1p4

[![Build Status](http://103.68.223.207:8111/app/rest/builds/buildType:(id:KcgMlSd1p4_Build)/statusIcon)](http://teamcity/viewType.html?buildTypeId=KcgMlSd1p4_Build&guest=1)

## Summary

- [kcg-ml-sd1p4](#kcg-ml-sd1p4)
   - [Summary](#summary)
   - [Downloading models](#downloading-models)
   - [Install requirements](#install-requirements)
   - [Running stable diffusion scripts](#running-stable-diffusion-scripts)
      - [Text to image](#text-to-image)
      - [Images from noise](#images-from-noise)
      - [Image from a list of prompts](#image-from-a-list-of-prompts)
      - [Image from another image](#image-from-another-image)
      - [Inpainting an image](#inpainting-an-image)
   - [Notebooks](#notebooks)
      - [Cleaning Jupyter Notebooks for Version Control](#cleaning-jupyter-notebooks-for-version-control)
      - [Installation](#installation)
      - [Setting up nbstripout](#setting-up-nbstripout)
      - [Using nbconvert](#using-nbconvert)
- [Generate images from prefixed noise vectors/seeds](#generate-images-from-prefixed-noise-vectorsseeds)
   - [Usage](#usage)

## Downloading models

```bash
./download-model.sh
```

## Install requirements

Install required dependency by running
```
pip install -r requirements.txt
```

## Running stable diffusion scripts

### Text to image

`scripts/text_to_image.py` creates an image from a prompt.

**Example**

```bash
python3 ./scripts/text_to_image.py --prompt "a girl by the lake staring at the stars"
```

**Command line arguments**

- `--prompt`: Prompt to generate the image from. Defaults to _'a painting of a cute monkey playing guitar'_.
- `--batch_size`: Batch size to use. Defaults to _4_.
- `--output`: Output path to store the generated images. Defaults to _'./outputs'_.
- `--sampler`: Set the sampler. Defaults to _'ddim'_. Options are _'ddim'_ and _'ddpm'_.
- `--checkpoint_path`: Relative path of the checkpoint file (*.ckpt). Defaults to _'./sd-v1-4.ckpt'_.
- `--flash`: Whether to use flash attention. Defaults to _False_.
- `--steps`: Number of sampling steps. Defaults to _50_.
- `--scale`: Unconditional guidance scale: **_eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))_** Defaults to _5.0_.
- `--low_vram`: Reduce VRAM usage. Defaults to _False_.
- `--force_cpu`: Force the use of CPU. Defaults to _False_.
- `--cuda-device`: CUDA device to use. Defaults to _cuda:0_.

### Images from noise

`scripts/generate_images_from_noise.py` creates a number of images (`--num_seeds`) for each of the seeds listed in a file (`--artist_file`).

**Example**

``` bash
python3 ./scripts/generate_images_from_noise.py
```

**Command line arguments**

- `--prompt_prefix`: Prefix for the prompt, must end with "in the style of". Default value="A woman with flowers in her hair in a courtyard, in the style of".
- `--artist_file`:  Path to the file containing the artists, each on a line. Defaults to _'../input/prompts/artists.txt'_.
- `--output`: Path to the output directory. Defaults to _'./outputs'_.
- `--checkpoint_path`: Path to the checkpoint file. Defaults to _'./input/model/sd-v1-4.ckpt'_.
- `--sampler`: Name of the sampler to use. Defaults to _'ddim'. Options are 'ddim' and 'ddpm'_.
- `--steps`: Number of steps to use. Defaults to _20_.
- `--num_seeds`: Number of seeds to use. Defaults to _8_.
- `--noise_file`: Path to the file containing the noise seeds, each on a line. Defaults to _'noise-seeds.txt'_.

### Image from a list of prompts

`scripts/image_from_prompt_list.py` creates a number of images (`--num_images`) for each of the prompts listed in a prompts file (`--prompts_file`). 

**Example**

``` bash
python3 ./scripts/image_from_prompt_list.py
```

**Command line arguments**

- `--num_images`: Number of images to generate per prompt. Defaults to _4_.
- `--checkpoint_path`: Path to the model. Defaults to _'./input/model/sd-v1-4.ckpt'_.
- `--prompts_file`: Path to the file containing the prompts, each on a line. Defaults to _'./input/prompts.txt'_.
- `--output`: Path to the output directory. Defaults to _'./outputs'_.

### Image from another image

`scripts/image_to_image.py` creates an image from another image (`--orig_img`) and a prompt (`--prompt`).  

**Example**

``` bash
python3 ./scripts/image_to_image.py --orig_img ./input/orig-images/monkey.jpg --prompt "a painting of a cute monkey playing guitar"
```

**Command line arguments**

- `--prompt`: Prompt to generate the image from. Defaults to _'a painting of a cute monkey playing guitar'_.
- `--orig_img`: Path to the original image. MANDATORY.
- `--batch_size`: Batch size to use. Defaults to _4_.
- `--steps`: Number of steps to use. Defaults to _50_.
- `--scale`: Unconditional guidance scale. Defaults to _5.0_.
- `--strength`: Strength for noise. Defaults to _0.75_.
- `--checkpoint_path`: Path to the checkpoint file. Defaults to _'./input/model/sd-v1-4.ckpt'_.
- `--output`: Path to the output directory. Defaults to _'./outputs'_.
- `--force_cpu`: Force the use of CPU. Defaults to _False_.
- `--cuda-device`: CUDA device to use. Defaults to _cuda:0_.

### Inpainting an image

`scripts/in_paint.py` performs inpainting based on a prompt (`--prompt`) to an image (`--orig_img`).

**Example**

``` bash
python3 ./scripts/in_paint.py --orig_img ./input/orig-images/monkey.jpg --prompt "a painting of a cute monkey playing guitar"
```

**Command line arguments**

- `--prompt`: Prompt to generate the image from. Defaults to _'a painting of a cute monkey playing guitar'_.
- `--checkpoint_path`: Path to the checkpoint file. Defaults to _'./input/model/sd-v1-4.ckpt'_.
- `--orig_img`: Path to the input image. MANDATORY.
- `--output`: Path to the output directory. Defaults to _'./outputs'_.
- `--batch_size`: Batch size to use. Defaults to _4_.
- `--steps`: Number of steps to use. Defaults to _50_.
- `--scale`: Unconditional guidance scale. Defaults to _5.0_.
- `--strength`: Strength for noise. Defaults to _0.75_.
- `--force_cpu`: Force the use of CPU. Defaults to _False_.
- `--cuda-device`: CUDA device to use. Defaults to _cuda:0_.

## Notebooks
| Notebook Title | Google Colab Link |
| --- | --- |
| Diffusers Unit Test Example | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kk-digital/kcg-ml-sd1p4/blob/main/notebooks/diffusers_unit_test.ipynb)|

### Cleaning Jupyter Notebooks for Version Control

### Installation

First, make sure you have nbstripout and nbconvert installed . You can install them using pip:

```sh
pip install nbstripout nbconvert
```


### Setting up nbstripout

```sh
nbstripout --install
```
Alternative installation to git attributes
```sh
nbstripout --install --attributes .gitattributes
```
### Using nbconvert
```sh
python -m nbconvert --ClearOutputPreprocessor.enabled=True --to notebook *.ipynb --inplace
```

# Generate images from prefixed noise vectors/seeds
The generate_noise_image.py generates noise seeds, saves them to a file, and uses a file called "artists.txt" in order to generate prompts using a set prefix. It generates an image for each noise seed + artist + prefix combination.
## Usage
```bash
python3 scripts/generate_noise_images.py \
	--prompt_prefix {enter-the-prefix-for-prompt-here} \
	--artist_file {default: input/artists.txt} \
	--output_dir {default: output/noise-tests} \
	--checkpoint_path {path-to-your-stable-diffusion-model-ckpt-file} \
	--sampler_name {sampler-here | default: ddim} \
	--n_steps {steps-for-image-generation} \
	--num_seeds {how-many-seeds-variations-for-each-prompt} \
	--noise_file {path-to-the-file-were-noise-seeds-are-going-to-be-stored}
```

The images will be stored on the output_dir, with the following name:
```
[artist-number-in-file]_n[noise-seed].jpg
```
