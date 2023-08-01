import os
import sys
import argparse
import torch
from os.path import join

sys.path.insert(0, os.getcwd())
from stable_diffusion.model.clip_text_embedder import CLIPTextEmbedder
from stable_diffusion.utils_backend import get_device



EMBEDDED_PROMPTS_DIR = os.path.abspath("./input/embedded_prompts/")

parser = argparse.ArgumentParser("Embed prompts using CLIP")
parser.add_argument(
    "-p",
    "--prompts",
    nargs="+",
    type=str,
    default=["A painting of a computer virus", "An old photo of a computer scientist"],
    help="The prompts to embed. Defaults to ['A painting of a computer virus', 'An old photo of a computer scientist']",
)
parser.add_argument(
    "--embedded_prompts_dir",
    type=str,
    default=EMBEDDED_PROMPTS_DIR,
    help="The path to the directory containing the embedded prompts tensors. Defaults to a constant EMBEDDED_PROMPTS_DIR, which is expected to be './input/embedded_prompts/'",
)
args = parser.parse_args()

NULL_PROMPT = ""
PROMPTS = args.prompts

os.makedirs(EMBEDDED_PROMPTS_DIR, exist_ok=True)

if __name__ == "__main__":
    null_prompt = NULL_PROMPT
    prompts = PROMPTS

    clip_text_embedder = CLIPTextEmbedder(device=get_device())
    clip_text_embedder.load_submodels()

    null_cond = clip_text_embedder(null_prompt)
    torch.save(null_cond, join(EMBEDDED_PROMPTS_DIR, "null_cond.pt"))
    print(
        "Null prompt embedding saved at: ",
        f"{join(EMBEDDED_PROMPTS_DIR, 'null_cond.pt')}",
    )

    embedded_prompts = clip_text_embedder(prompts)
    # print("embedded_prompts shape: ", embedded_prompts.shape)
    torch.save(embedded_prompts, join(EMBEDDED_PROMPTS_DIR, "embedded_prompts.pt"))
    print(
        "Prompts embeddings saved at: ",
        f"{join(EMBEDDED_PROMPTS_DIR, 'embedded_prompts.pt')}",
    )
