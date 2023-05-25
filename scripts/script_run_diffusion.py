import os, sys
import hashlib
import argparse
import subprocess
from text_to_image import Txt2Img
import torch
from tqdm import tqdm

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num_images', type=int, default=10, help='Number of images to generate')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
args = parser.parse_args()

# Set the output directory
output_dir = '/output'

# Check if CUDA is available
if not torch.cuda.is_available():
    print("WARNING: You are running this script without CUDA. Brace yourself for a slow ride.")

# Load the prompts
prompts_path = '/input/prompts.txt'
with open(prompts_path, 'r') as f:
    prompts = f.readlines()

# Load the Stable Diffusion model
checkpoint_path = os.path.join(os.path.dirname(__file__), '/input/model/sd-v1-4.ckpt')
sampler_name = 'ddim'
txt2img = Txt2Img(checkpoint_path=checkpoint_path, sampler_name=sampler_name, n_steps=20)

# Generate the images
batch_size = args.batch_size
for prompt in prompts:
    hash = hashlib.sha1(prompt.encode("UTF-8")).hexdigest()
    filename = hash[:10] + ".jpg"
    dest_path = os.path.join(output_dir, filename)
    # Check if the image already exists
    if not os.path.isfile(dest_path):
        txt2img(dest_path=dest_path, batch_size=batch_size, prompt=prompt)

# Unload model
del txt2img
print("done!")
