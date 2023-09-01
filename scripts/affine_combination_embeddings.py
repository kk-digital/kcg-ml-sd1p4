from PIL import Image

import argparse
import os
import sys

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from stable_diffusion.utils_backend import get_device
from ga.prompt_generator import generate_prompts
def parse_arguments():
    """Command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Affine combination of embeddings.")

    parser.add_argument('--output', type=str, help='Output folder')
    parser.add_argument('--image_width', type=int, default=64)
    parser.add_argument('--image_height', type=int, default=64)
    parser.add_argument('--cfg_strength', type=float, default=12)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--sampler', type=str, default='ddim')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    output = args.output
    image_width = args.image_width
    image_height = args.image_height
    cfg_strength = args.cfg_strength
    device = get_device(args.device)
    sampler = args.sampler
    num_prompts = args.num_prompts
    num_phrases = args.num_phrases

    # Generate N Prompts
    prompt_list = generate_prompts(num_prompts, num_phrases)
    # Get N Embeddings

    # Combinate into one Embedding
    # Maximize fitness
