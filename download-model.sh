#!/bin/bash

# Check if the model ckpt file exists
if [ ! -f "/stable_diffusion/modelstore/sd-v1-4.ckpt" ]; then
    # Download the model ckpt file
    wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt -O /stable_diffusion/modelstore/sd-v1-4.ckpt
fi
