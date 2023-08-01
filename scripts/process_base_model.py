"""
---
title: Utility functions for stable diffusion
summary: >
 Utility functions for stable diffusion
---

# initializes and saves all models
1 saves clip embedder
2 saves autoencoder
3 saves unet
0 saves all submodels
else saves latent diffusion model with state dict loaded from checkpoint
"""

import os
import sys
import argparse

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.labml.monit import section
from stable_diffusion.constants import CHECKPOINT_PATH
from stable_diffusion.constants import ROOT_MODELS_DIR, ROOT_MODELS_PREFIX
from stable_diffusion.utils_model import (
    initialize_latent_diffusion,
)
from stable_diffusion.model.clip_image_encoder import CLIPImageEncoder

try:
    from torchinfo import summary
except:
    print("torchinfo not installed")
    summary = lambda x: print(x)

parser = argparse.ArgumentParser(description="")

parser.add_argument("checkpoint_path", type=str, default=CHECKPOINT_PATH)
parser.add_argument("--use_ckpt", type=int, default=False)
parser.add_argument("-g", "--granularity", type=int, default=0)
parser.add_argument("--root_models_dir", type=str, default=ROOT_MODELS_DIR)

args = parser.parse_args()

# print(args)
GRANULARITY = args.granularity
CHECKPOINT_PATH_ARG = args.checkpoint_path
ROOT_MODELS_DIR = args.root_models_dir
IMAGE_ENCODER = True
USE_CKPT = args.use_ckpt

if USE_CKPT and not ".ckpt" in CHECKPOINT_PATH_ARG:
    print("WARNING: .ckpt used but CHECKPOINT_PATH does not contain .ckpt")
    print("Defaulting to configured default .ckpt checkpoint path")
    CHECKPOINT_PATH_ARG = CHECKPOINT_PATH


def create_folder_structure(root_dir: str = ROOT_MODELS_PREFIX) -> None:
    embedder_submodels_folder = os.path.abspath(
        os.path.join(root_dir, "clip_text_embedder/")
    )
    os.makedirs(embedder_submodels_folder, exist_ok=True)

    image_encoder_submodels_folder = os.path.abspath(
        os.path.join(root_dir, "clip_image_encoder/")
    )
    os.makedirs(image_encoder_submodels_folder, exist_ok=True)

    autoencoder_submodels_folder = os.path.abspath(
        os.path.join(root_dir, "autoencoder/")
    )
    os.makedirs(autoencoder_submodels_folder, exist_ok=True)

    unet_submodels_folder = os.path.abspath(os.path.join(root_dir, "unet/"))
    os.makedirs(unet_submodels_folder, exist_ok=True)

    latent_diffusion_submodels_folder = os.path.abspath(
        os.path.join(root_dir, "latent_diffusion/")
    )
    os.makedirs(latent_diffusion_submodels_folder, exist_ok=True)


if __name__ == "__main__":
    create_folder_structure(root_dir=ROOT_MODELS_DIR)

    model = initialize_latent_diffusion(
        path=CHECKPOINT_PATH_ARG, force_submodels_init=True
    )

    summary(model)

    if GRANULARITY == 0:
        if IMAGE_ENCODER:
            with section(
                    "initialize CLIP image encoder and load submodels from lib"
            ):
                img_encoder = CLIPImageEncoder()
                img_encoder.load_from_lib()
            with section("save image encoder submodels"):
                img_encoder.save_submodels()
                img_encoder.unload_submodels()
                img_encoder.save()
        model.first_stage_model.save()
        with section("save vae submodels"):
            model.first_stage_model.save_submodels()  # saves autoencoder submodels (encoder, decoder) with loaded state dict
        with section("save embedder submodels"):
            model.cond_stage_model.save_submodels()  # saves text embedder submodels (tokenizer, transformer) with loaded state dict
        with section("save latent diffusion submodels"):
            model.save_submodels()  # saves latent diffusion submodels (autoencoder, clip_embedder, unet) with loaded state dict
        with section("save latent diffusion model"):
            model.save()  # saves latent diffusion model with loaded state dict and unloaded submodels
        with section("unload vae submodels"):
            model.first_stage_model.unload_submodels()  # unloads autoencoder submodels
        with section("unload embedder submodels"):
            model.cond_stage_model.unload_submodels()  # unloads text embedder submodels
        with section("unload latent diffusion submodels"):
            model.unload_submodels()  # unloads latent diffusion submodels
    elif GRANULARITY == 1:
        if IMAGE_ENCODER:
            with section(
                    "initialize CLIP image encoder and load submodels from lib"
            ):
                img_encoder = CLIPImageEncoder()
                img_encoder.load_from_lib()
            with section("save image encoder"):
                img_encoder.save()
                img_encoder.unload_submodels()
        with section("save latent diffusion submodels"):
            model.save_submodels()  # saves latent diffusion submodels (autoencoder, clip_embedder and unet) with loaded state dict loaded submodels
        with section("save latent diffusion model"):
            model.save()  # saves latent diffusion model with loaded state dict and unloaded submodels
        with section("unload latent diffusion submodels"):
            model.unload_submodels()  # unloads latent diffusion submodels
    elif GRANULARITY == 2:
        with section("save latent diffusion model"):
            model.save()  # saves latent diffusion model with loaded state dict and loaded submodels