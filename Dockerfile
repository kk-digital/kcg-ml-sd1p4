# Use base image
FROM docker.io/nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Set working directory
WORKDIR /stable_diffusion

# Mount volumes
VOLUME /input
VOLUME /output

# Set environment variable to avoid interactive configuration prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    curl \
    transmission-cli \
    libboost-python-dev \
    libboost-system-dev \
    libboost-chrono-dev \
    libtorrent-rasterbar-dev \
    git \
    && rm -rf /var/lib/apt/lists/*
# Copy files
COPY /stable_diffusion/requirements.txt /stable_diffusion/requirements.txt
COPY /stable_diffusion/script_run_diffusion.py /stable_diffusion/script_run_diffusion.py

# These deps are large, so put them in their own layer to save rebuild time
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu116 \
    diffusers==0.6.0 \
    torch==1.12.1+cu116

# Install Python dependencies from requirements.txt
RUN pip3 install -r requirements.txt

# Install python-qbittorrent
RUN pip3 install python-qbittorrent

# Create a directory for the models
RUN mkdir /input/models

# Set environment variable for log directory
ENV LOG_DIR=/output/logs


# Download the model if it doesn't exist

# Run the main command with logging and stats
CMD echo "Start Time: $(date)" \
    && python3 script_run_diffusion.py \
    && echo "End Time: $(date)" \
