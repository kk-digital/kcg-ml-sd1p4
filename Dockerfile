# Use base image
FROM docker.io/nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# These deps are large, so put them in their own layer to save rebuild time
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN pip3 install --upgrade pip
RUN pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113


# Mount volumes
VOLUME /input
VOLUME /output
VOLUME /stable_diffusion


# Set environment variable to avoid interactive configuration prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    libboost-python-dev \
    libboost-system-dev \
    libboost-chrono-dev \
    libtorrent-rasterbar-dev \
    git \
    && rm -rf /var/lib/apt/lists/*
# Change directory
WORKDIR /tmp/

# Copy files
COPY /stable_diffusion/requirements.txt /tmp/
COPY /stable_diffusion/script_run_diffusion.py /tmp/

# Install Python dependencies from requirements.txt
RUN pip3 install --extra-index-url https://download.pytorch.org/whl/cu116 -r /tmp/requirements.txt

# Set environment variable for log directory
ENV LOG_DIR=/output/logs

# Clone stable-diffusion repo
WORKDIR /
RUN git clone https://github.com/basujindal/stable-diffusion
RUN mv /stable-diffusion /repo
WORKDIR /repo

# Work around for VectorQuantizer2
RUN pip3 install taming-transformers-rom1504 clip kornia

# Set cache
RUN mkdir -p /root/.cache/huggingface/
RUN ln -sf /stable_diffusion/modelstore/ /root/.cache/huggingface/transformers

# Run the main command with logging and stats
CMD echo "Start Time: $(date)" \
    && cd /stable_diffusion \
    && cp inference.py /repo/\
    && python3 /tmp/script_run_diffusion.py \
    && echo "End Time: $(date)" \
