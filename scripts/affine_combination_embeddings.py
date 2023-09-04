from PIL import Image

import argparse
import os
import sys
import numpy as np
import torch

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from stable_diffusion.utils_backend import get_device
from ga.prompt_generator import generate_prompts
from stable_diffusion.model.clip_text_embedder import CLIPTextEmbedder

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


def combine_embeddings(embeddings_array, weight_array):

    # empty embedding filled with zeroes
    result_embedding = np.zeros((77, 786))

    num_elements = len(embeddings_array)
    for i in range(num_elements):
        embedding = embeddings_array[i]
        weight = weight_array[i]

        embedding = embedding * weight
        result_embedding += embedding

    return result_embedding

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

    clip_text_embedder = CLIPTextEmbedder(device=get_device())
    clip_text_embedder.load_submodels()

    # Generate N Prompts
    prompt_list = generate_prompts(num_prompts, num_phrases)

    embedded_prompts_array = []
    # Get N Embeddings
    for prompt in prompt_list:
        # get the embedding from positive text prompt
        embedded_prompts = clip_text_embedder(prompt.positive_prompt_str)
        embedded_prompts_numpy = embedded_prompts.cpu().numpy()

        del embedded_prompts
        torch.cuda.empty_cache()

        embedded_prompts_array.append(embedded_prompts_numpy)

    # array of random weights
    weight_array = np.random.rand(num_prompts)
    # Combinate into one Embedding
    embedding = combine_embeddings(embedded_prompts_array, weight_array)
    # Maximize fitness
    print(embedding)