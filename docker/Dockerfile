# Use base image
FROM docker.io/nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Set environment variable to avoid interactive configuration prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libboost-system-dev \
    libboost-chrono-dev \
    git \
    && rm -rf /var/lib/apt/lists/*


# Install dependencies
ARG DEBIAN_FRONTEND=noninteractive

RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN --mount=type=cache,target=/root/.cache/pip pip3 install \
	torch==1.11.0+cu113 \
	torchvision==0.12.0+cu113 \
	torchaudio==0.11.0 \
	taming-transformers-rom1504 \
	clip \
	kornia \
	--extra-index-url https://download.pytorch.org/whl/cu113

# Install other dependencies
COPY ../requirements.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/pip pip3 install \
	-r /tmp/requirements.txt

# Mount volumes
VOLUME input output stable_diffusion

# Set environment variable for log directory
ENV LOG_DIR=/output/logs

# Clone stable-diffusion repo
WORKDIR /
RUN git clone https://github.com/basujindal/stable-diffusion \
	&& mv /stable-diffusion /repo
WORKDIR /repo

# Setup model cache
RUN mkdir -p /root/.cache/huggingface
RUN ln -sf /input/model /root/.cache/huggingface/
RUN mv /root/.cache/huggingface/model /root/.cache/huggingface/transformers

# Run the main command with logging and stats
CMD echo "Start Time: $(date)" \
    && chmod -R 777 /input/model \
    && cp /stable_diffusion/inference.py /repo/inference.py \
    && cp /stable_diffusion/script_run_diffusion.py /tmp \
    && python3 /tmp/script_run_diffusion.py \
    && echo "End Time: $(date)" \
