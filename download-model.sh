#!/bin/bash

# Make modelstore directory if not there
mkdir ./stable_diffusion/modelstore

# Check if the model ckpt file exists
if [ ! -f "./stable_diffusion/modelstore/sd-v1-4.ckpt" ]; then
    # Download the model ckpt file
    wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt -O ./input/models/sd-v1-4.ckpt
fi

#if [ ! -f "./stable_diffusion/modelstore/clip.bin" ]; then
    # Download the model clip file
#    wget https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin -O ./stable_diffusion/modelstore/clip.bin
#fi
