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

# Create a directory for the models and download the model using BitTorrent
RUN mkdir /models \
    && cd /models \
    && transmission-cli "magnet:?xt=urn:btih:3f8016061132ad4a475eb33dac569f3a88418974&dn=v1-5-pruned-emaonly.safetensors&tr=udp%3a%2f%2fopen.demonii.com%3a1337%2fannounce&tr=udp%3a%2f%2fexodus.desync.com%3a6969%2fannounce" \
    && rm -rf /var/lib/apt/lists/*

# Create volume for input and output directories from host
VOLUME ["/host/tmp/input", "/host/tmp/output"]

# Set environment variable for log directory
ENV LOG_DIR=/app/output/logs

# Run the main command with logging and stats
CMD echo "Start Time: $(date)" \
    && transmission-cli "magnet:?xt=urn:btih:3f8016061132ad4a475eb33dac569f3a88418974&dn=v1-5-pruned-emaonly.safetensors&tr=udp%3a%2f%2fopen.demonii.com%3a1337%2fannounce&tr=udp%3a%2f%2fexodus.desync.com%3a6969%2fannounce" \
    && echo "End Time: $(date)" \
    && echo "M/S: $(du -sh /models)" \
    && echo "Stats: $(transmission-show -s /models/*.torrent)"
