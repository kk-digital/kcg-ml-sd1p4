# Use base image
FROM docker.io/pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set environment variable to avoid interactive configuration prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    libboost-system-dev \
    libboost-chrono-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Add stable diffusion repo
COPY  ./ /repo
WORKDIR /repo

#RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# local pip cache
RUN pip3 config set global.index-url http://192.168.3.2:5010/index/
RUN pip3 config set global.trusted-host 192.168.3.2

# Install other dependencies
RUN --mount=type=cache,target=/root/.cache/pip pip3 install \
	-r /repo/requirements.txt

# Set environment variable for log directory
ENV LOG_DIR=/output/logs

# Setup model cache
RUN mkdir -p /root/.cache/huggingface
RUN ln -sf /input/model /root/.cache/huggingface

# Run the main command with logging and stats
CMD echo "Start Time: $(date)" \
    && chmod -R 777 /repo/input/model \
    && python3 /repo/scripts/script_run_diffusion.py \
    && echo "End Time: $(date)" \
