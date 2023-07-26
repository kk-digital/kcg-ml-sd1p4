#!/bin/bash

# Make model directory if not there
mkdir -p ./input/model

# Check if the model file exists
if [ ! -f "./input/model/v1-5-pruned-emaonly.safetensors" ]; then
    # Download the model file
    wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors -O ./input/model/v1-5-pruned-emaonly.safetensors
fi