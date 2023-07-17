import os
import sys
import argparse

sys.path.insert(0, os.getcwd())

from typing import List
import torch

from os.path import join

from stable_diffusion2.model.clip_text_embedder import CLIPTextEmbedder
from stable_diffusion2.utils.utils import check_device, get_memory_status


EMBEDDED_PROMPTS_DIR = os.path.abspath("./input/embedded_prompts/")

parser = argparse.ArgumentParser("Embed prompts using CLIP")
parser.add_argument(
    "-p",
    "--prompts",
    nargs="+",
    type=str,
    default=["An old photo of a computer scientist"],
    help="The prompts to embed. Defaults to ['An old photo of a computer scientist']",
)
parser.add_argument(
    "--embedded_prompts_dir",
    type=str,
    default=EMBEDDED_PROMPTS_DIR,
    help="The path to the directory containing the embedded prompts tensors. Defaults to a constant EMBEDDED_PROMPTS_DIR, which is expected to be './input/embedded_prompts/'",
)
parser.add_argument(
    "--num_images",
    type=str,
    default=10,
    help="The number of images to generate",
)
args = parser.parse_args()

NULL_PROMPT = ""
PROMPTS = args.prompts
NUM_IMAGES = args.num_images

os.makedirs(EMBEDDED_PROMPTS_DIR, exist_ok=True)

def embed_and_save_prompts(prompts: list, null_prompt = NULL_PROMPT):

    null_prompt = null_prompt
    prompts = prompts

    clip_text_embedder = CLIPTextEmbedder(device=check_device())
    clip_text_embedder.load_submodels()

    null_cond = clip_text_embedder(null_prompt)
    torch.save(null_cond, join(EMBEDDED_PROMPTS_DIR, "null_cond.pt"))
    print(
        "Null prompt embedding saved at: ",
        f"{join(EMBEDDED_PROMPTS_DIR, 'null_cond.pt')}",
    )

    embedded_prompts = clip_text_embedder(prompts)
    torch.save(embedded_prompts, join(EMBEDDED_PROMPTS_DIR, "embedded_prompts.pt"))
    
    print(
        "Prompts embeddings saved at: ",
        f"{join(EMBEDDED_PROMPTS_DIR, 'embedded_prompts.pt')}",
    )
    
    get_memory_status()
    clip_text_embedder.to("cpu")
    del clip_text_embedder
    torch.cuda.empty_cache()
    get_memory_status()
    return embedded_prompts, null_cond

if __name__ == "__main__":

    embedded_prompts, null_cond = embed_and_save_prompts(PROMPTS)


