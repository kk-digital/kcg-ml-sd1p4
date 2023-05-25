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
    - [Download the model](#download-the-model)
    - [Create the Docker image](#create-the-docker-image)


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
### Download the model
```bash
./download-model.sh
```
### Create the Docker image

```bash
export DOCKER_BUILDKIT=1
docker build . --file ./docker/Dockerfile --tag stable-diffusion
docker run --gpus all \
	-v input:/input
	-v output:/output \
	stable-diffusion
```
