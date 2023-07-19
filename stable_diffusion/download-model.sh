#!/bin/bash

# Make model directory if not there
mkdir -p ./input/model/

# Check if the model ckpt file exists
if [ ! -f "./input/model/v1-5-pruned-emaonly.ckpt" ]; then
    # Download the model ckpt file
    wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt -O ./input/model/v1-5-pruned-emaonly.ckpt
fi
