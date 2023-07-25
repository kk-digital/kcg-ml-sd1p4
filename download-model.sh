#!/bin/bash

# Make model directory if not there
mkdir -p ./input/model

# Check if transmission client is installed
if ! type "transmission-cli" > /dev/null; then
  # Direct the user to install transmission CLI
  printf "\ntransmission-cli not found! \nPlease install transmission-cli using your distro's package manager."
  exit 1
fi

# Check if the model file exists
if [ ! -f "./input/model/v1-5-pruned-emaonly.safetensors" ]; then
    # Download the model file
    transmission-cli --download-dir ./input/model/v1-5-pruned-emaonly.safetensors "magnet:?xt=urn:btih:H6ABMBQRGKWUUR26WM62YVU7HKEEDCLU&dn=v1-5-pruned-emaonly.safetensors&xl=4265146304&tr=udp%3A%2F%2Fopen.stealth.si%3A80%2Fannounce"
fi
