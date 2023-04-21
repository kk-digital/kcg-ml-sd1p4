import os
import argparse
import subprocess
import torch
import numpy as np
from tqdm import tqdm

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_images', type=int, default=10, help='Number of images to generate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    args = parser.parse_args()

    # Load the diffusion model
    model_path = '/input/model/sd-v1-4.ckpt'

    # Load the prompts
    prompts_path = '/input/prompts.txt'
    with open(prompts_path, 'r') as f:
        prompts = f.readlines()

    # Generate the images
    device = torch.device('cuda')
    batch_size = args.batch_size
    for i in tqdm(range(0, args.num_images, batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        cmd = ['python3', '/repo/inference.py','--prompt'] + batch_prompts
        print(cmd)
        subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
