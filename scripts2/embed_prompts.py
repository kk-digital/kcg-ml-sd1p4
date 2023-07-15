import os
import sys
import argparse
sys.path.insert(0, os.getcwd())

from typing import List
import torch

from os.path import join

from stable_diffusion2.model.clip_text_embedder import CLIPTextEmbedder
from stable_diffusion2.utils.utils import check_device


OUTPUT_DIR = './input/embedded_prompts/'

parser = argparse.ArgumentParser("Embed prompts using CLIP")
parser.add_argument('-p', '--prompts', nargs = "+", type=str, default=["A painting of a computer virus", "An old photo of a computer scientist"], help = "The prompts to embed. Defaults to ['A painting of a computer virus', 'An old photo of a computer scientist']")
args = parser.parse_args()

NULL_PROMPT = ""
PROMPTS = args.prompts

os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    
    null_prompt = NULL_PROMPT
    prompts = PROMPTS

    clip_text_embedder = CLIPTextEmbedder(device = check_device())
    clip_text_embedder.load_submodels()

    null_cond = clip_text_embedder(null_prompt)
    print("null_cond shape: ", null_cond.shape)
    torch.save(null_cond, join(OUTPUT_DIR, "null_cond.pt"))

    embedded_prompts = clip_text_embedder(prompts)
    print("embedded_prompts shape: ", embedded_prompts.shape)
    torch.save(embedded_prompts, join(OUTPUT_DIR, "embedded_prompts.pt"))
