#!/bin/bash

# Make model directory if not there
mkdir -p ./input/model

# Check if the model ckpt file exists
if [ ! -f "./input/models/sd-v1-4.ckpt" ]; then
    # Download the model ckpt file
    wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt -O ./input/model/sd-v1-4.ckpt
fi
