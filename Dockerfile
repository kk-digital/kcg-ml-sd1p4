FROM python:3.10
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04


RUN sudo apt-get update 
RUN sudo apt-get install -y python3-pip
RUN sudo apt-get install -y python3-dev
RUN sudo apt-get install -y wget



#installing cuda
RUN wget https://developer.download.nvidia.com/compute/cuda/11.4.2/local_installers/cuda-repo-ubuntu2004-11-4-local_11.4.2-470.42.01-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2004-11-4-local_11.4.2-470.42.01-1_amd64.deb
RUN apt-key add /var/cuda-repo-ubuntu2004-11-4-local/7fa2af80.pub
RUN apt-get update
RUN apt-get -y install cuda
#attaching cuda to path

RUN apt-get install -y wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

RUN wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckptipts/data/stable_diffusion/
RUN mv sd-v1-4.ckpt /app/stable_diffusion/scripts/data/stable_diffusion/


# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# Runs the application
CMD ["python", "stable_diffusion/scripts/text_to_image.py"]
