# Use base image
FROM docker.io/nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# These deps are large, so put them in their own layer to save rebuild time
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN pip install --extra-index-url https://download.pytorch.org/whl/cu116 \
    diffusers==0.6.0 \
    torch==1.12.1+cu116

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
# Copy files
WORKDIR /stable_diffusion

# Install Python dependencies from requirements.txt
RUN pip3 install -r requirements.txt

# Set environment variable for log directory
ENV LOG_DIR=/output/logs

# Run the main command with logging and stats
CMD echo "Start Time: $(date)" \
    && python3 script_run_diffusion.py \
    && echo "End Time: $(date)" \
