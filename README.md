# kcg-ml-sd1p4

[![Build Status](http://103.68.223.207:8111/app/rest/builds/buildType:(id:KcgMlSd1p4_Build)/statusIcon)](http://teamcity/viewType.html?buildTypeId=KcgMlSd1p4_Build&guest=1)

## Summary

- [kcg-ml-sd1p4](#kcg-ml-sd1p4)
  - [Summary](#summary)
  - [Downloading Models](#downloading-models)
  - [Install requirements](#install-requirements)
  - [Running stable diffusion model](#running-stable-diffusion-model)
  - [Scripts](#scripts)
    - [Install requirements](#install-requirements-1)
  - [Notebooks](#notebooks)
    - [Cleaning Jupyter Notebooks for Version Control](#cleaning-jupyter-notebooks-for-version-control)
    - [Installation](#installation)
    - [Setting up nbstripout](#setting-up-nbstripout)
    - [Using nbconvert](#using-nbconvert)
- [Setting up Docker for the Stable Diffusion server](docker/README.md)
  - [Generating images from different noise vectors](#generate-images-from-prefixed-noise-vectors/seeds)
    - [Usage](#usage)


## Downloading Models

```bash
./download-model.sh
```

## Install requirements

Install required dependency by running
```
pip install -r requirements.txt
```

## Running stable diffusion model

You can run the stable diffusion model with:
```
python3 ./scripts/text_to_image.py --prompt "a girl by the lake staring at the stars"

Note:
-put-out /  --outdir "./output/"
```
You can try your own prompt by replacing "test" with "your prompt"


## Scripts

To run the scripts, you must install the requirements:

### Install requirements

Install required dependency by running
```
pip install -r requirements.txt
```

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
