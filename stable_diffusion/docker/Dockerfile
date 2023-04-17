# Use base image
FROM docker.io/nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Set working directory
WORKDIR /app

# Copy script.py and requirements.txt into the container
COPY script.py /app/script.py
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    curl \
    transmission-cli \
    && rm -rf /var/lib/apt/lists/*

# Run tests for the script
RUN python3 /app/script.py test

# Create a directory for the models
RUN mkdir /models

# Set environment variable for log directory
ENV LOG_DIR=/app/output/logs

# Run the main command with logging and stats
CMD echo "Start Time: $(date)" \
    && python3 /app/script.py \
    && echo "End Time: $(date)" \
    && echo "M/S: $(du -sh /models)" \
    && echo "Stats: $(transmission-show -s /models/*.torrent)" 
