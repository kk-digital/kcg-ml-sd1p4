# kcg-ml-sd1p4

[![Build Status](http://103.68.223.207:8111/app/rest/builds/buildType:(id:KcgMlSd1p4_Build)/statusIcon)](http://teamcity/viewType.html?buildTypeId=KcgMlSd1p4_Build&guest=1)

## Summary

- [kcg-ml-sd1p4](#kcg-ml-sd1p4)
  - [Summary](#summary)
  - [Learn how to use the scripts](#learn-how-to-use-the-scripts)
  - [Scripts](#scripts)
    - [Install requirements](#install-requirements)
  - [Notebooks](#notebooks)
    - [Cleaning Jupyter Notebooks for Version Control](#cleaning-jupyter-notebooks-for-version-control)
    - [Installation](#installation)
    - [Setting up nbstripout](#setting-up-nbstripout)
    - [Using nbconvert](#using-nbconvert)
- [Setting up Docker for the Stable Diffusion server](#setting-up-docker-for-the-stable-diffusion-server)
  - [Installing Docker](#installing-docker)
    - [Add the docker GPG key to the repository](#add-the-docker-gpg-key-to-the-repository)
    - [Add docker to apt sources](#add-docker-to-apt-sources)
    - [Update the package database](#update-the-package-database)
    - [Install docker](#install-docker)
    - [Startup docker](#startup-docker)
    - [Install the NVIDIA toolkit (for Ubuntu)](#install-the-nvidia-toolkit-for-ubuntu)
    - [Verify the NVIDIA container toolkit is working](#verify-the-nvidia-container-toolkit-is-working)
    - [Add yourself to the docker group](#add-yourself-to-the-docker-group)
    - [Verify you can run docker commands](#verify-you-can-run-docker-commands)
    - [Create input and output directories](#create-input-and-output-directories)
    - [Download the model](#download-the-model)
    - [Create the Docker image](#create-the-docker-image)
  - [Downloading Models](#downloading-models)

## Learn how to use the scripts

To learn how to use the scripts, please read the [README.md](DataUtility/README.md) file in the DataUtility folder.

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

# Setting up Docker for the Stable Diffusion server

## Installing Docker

### Add the docker GPG key to the repository
```bash
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```
### Add docker to apt sources
```bash
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
```
### Update the package database
```bash
sudo apt-get update
```
### Install docker
```bash
sudo apt-get install docker-ce docker-ce-cli containerd.io
```
### Startup docker
```bash
sudo systemctl enable docker --now
sudo systemctl status docker
```

### Install the NVIDIA toolkit (for Ubuntu)
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list && sudo apt-get update && sudo apt-get install -y nvidia-docker2 && sudo systemctl restart docker
```
### Verify the NVIDIA container toolkit is working
```bash
sudo docker run --gpus all nvidia/cuda:11.0-base nvidia-smi
```
This command should display information about the current GPUs on the system
### Add yourself to the docker group
```bash
sudo usermod -aG docker $USER
```
**You must log out and log back in to apply the changes**
### Verify you can run docker commands
```bash
docker run hello-world
```
### Create input and output directories
```bash
mkdir -p input/model output
```

### Download the model
```bash
./download-model.sh
```
### Create the Docker image

```bash
export DOCKER_BUILDKIT=1
docker build . --file ./docker/Dockerfile --tag stable-diffusion
docker run --gpus all -v ./input:/input -v ./output:/output -v ./stable_diffusion:/stable_diffusion stable-diffusion
```
Replace /path/to/input with your input path, and /path/to/output with your output that

## Downloading Models

```bash
./download-model.sh
```
