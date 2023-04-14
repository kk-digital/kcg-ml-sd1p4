FROM docker.io/nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

WORKDIR /app

COPY . /app

RUN apt update && \
    apt install -y python3-pip && \
    pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

EXPOSE 80

ENV NAME World


CMD ["python3", "image-generator.py"]


