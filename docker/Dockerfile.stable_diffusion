# Use base image
FROM docker.io/nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Set working directory
WORKDIR /app

# Copy script.py and requirements.txt into the container
COPY script_download_model.py /app/script_download_model.py
COPY requirements.txt /app/requirements.txt

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

# Install Python dependencies from requirements.txt
RUN pip3 install -r /app/requirements.txt

# Install python-qbittorrent
RUN pip3 install python-qbittorrent

# Create a directory for the models
RUN mkdir /models

# Set environment variable for log directory
ENV LOG_DIR=/app/output/logs

# Run the main command with logging and stats
CMD echo "Start Time: $(date)" \
    && python3 /app/script_download_model.py \
    && echo "End Time: $(date)" \
