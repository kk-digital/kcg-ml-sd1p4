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
