# Use base image
FROM docker.io/pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

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

# Clone stable-diffusion repo
WORKDIR /
RUN git clone https://github.com/kk-digital/kcg-ml-sd1p4/ \
	&& mv /kcg-ml-sd1p4 /repo
WORKDIR /repo

#RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
#RUN --mount=type=cache,target=/root/.cache/pip pip3 install \
#	torch==1.11.0+cu113 \
#	torchvision==0.12.0+cu113 \
#	torchaudio==0.11.0 \
#	taming-transformers-rom1504 \
#	clip \
#	kornia \
#	--extra-index-url https://download.pytorch.org/whl/cu113

# Install other dependencies
RUN --mount=type=cache,target=/root/.cache/pip pip3 install \
	-r /repo/requirements.txt

# Mount volumes
VOLUME input output stable_diffusion

# Set environment variable for log directory
ENV LOG_DIR=/output/logs

# Setup model cache
RUN mkdir -p /root/.cache/huggingface
RUN ln -sf /repo/input/model /root/.cache/huggingface

# Run the main command with logging and stats
CMD echo "Start Time: $(date)" \
    && chmod -R 777 /repo/input/model \
    && python3 /repo/stable_diffusion/script_run_diffusion.py \
    && echo "End Time: $(date)" \
