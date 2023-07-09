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

import random
import os
import argparse
from pathlib import Path
from typing import Union

import PIL
import numpy as np
import torch
from PIL import Image

# from labml import monit
# from labml.logger import inspect


from stable_diffusion2.constants import AUTOENCODER_PATH, ENCODER_PATH, DECODER_PATH
from stable_diffusion2.constants import EMBEDDER_PATH, TOKENIZER_PATH, TRANSFORMER_PATH
from stable_diffusion2.constants import UNET_PATH
from stable_diffusion2.constants import LATENT_DIFFUSION_PATH
from stable_diffusion2.constants import CHECKPOINT_PATH
from stable_diffusion2.constants import ROOT_MODELS_PATH

from stable_diffusion2.utils.model import initialize_autoencoder, initialize_clip_embedder, initialize_unet, initialize_latent_diffusion
from stable_diffusion2.utils.utils import SectionManager as section

from stable_diffusion2.latent_diffusion import LatentDiffusion
from stable_diffusion2.model.vae.encoder import Encoder
from stable_diffusion2.model.vae.decoder import Decoder
from stable_diffusion2.model.vae.autoencoder import Autoencoder
from stable_diffusion2.model.clip.clip_embedder import CLIPTextEmbedder
from stable_diffusion2.model.unet.unet import UNetModel

try:
    from torchinfo import summary
except:
    print('torchinfo not installed')
    summary = lambda x: print(x)

parser = argparse.ArgumentParser(
        description='')

parser.add_argument('--save_without_weights', type=bool, default=False)
parser.add_argument('--unet', type=bool, default=True)
parser.add_argument('--clip', type=bool, default=True)
parser.add_argument('--vae', type=bool, default=True)
parser.add_argument('--granularity', type=int, default=0)
parser.add_argument('--checkpoint_path', type=str, default=CHECKPOINT_PATH)
parser.add_argument('--root_models_path', type=str, default=ROOT_MODELS_PATH)

args = parser.parse_args()


# print(args)
GRANULARITY = args.granularity
CHECKPOINT_PATH = args.checkpoint_path
ROOT_MODELS_PATH = args.root_models_path
SAVE_WITHOUT_WEIGHTS = args.save_without_weights
SAVE_UNET = args.unet
SAVE_CLIP = args.clip
SAVE_VAE = args.vae


def create_folder_structure(root_dir: str = "./") -> None:
    
    embedder_submodels_folder = os.path.abspath(os.path.join(root_dir, 'clip/'))
    os.makedirs(embedder_submodels_folder, exist_ok=True)

    autoencoder_submodels_folder = os.path.abspath(os.path.join(root_dir, 'autoencoder/'))
    os.makedirs(autoencoder_submodels_folder, exist_ok=True)

    unet_submodels_folder = os.path.abspath(os.path.join(root_dir, 'unet/'))
    os.makedirs(unet_submodels_folder, exist_ok=True)

    latent_diffusion_submodels_folder = os.path.abspath(os.path.join(root_dir, 'latent_diffusion/'))
    os.makedirs(latent_diffusion_submodels_folder, exist_ok=True)

    
if __name__ == '__main__':
    create_folder_structure(root_dir=ROOT_MODELS_PATH)
    if SAVE_WITHOUT_WEIGHTS:
        if SAVE_CLIP:
            embedder = initialize_clip_embedder()
            summary(embedder)
            with section("to save submodels"):
                embedder.save_submodels()
            with section("to save embedder"):    
                embedder.save()
        if SAVE_VAE:
            autoencoder = initialize_autoencoder()
            summary(autoencoder)
            with section("to save submodels"):
                autoencoder.save_submodels()
            with section("to save autoencoder"):
                autoencoder.save()
        if SAVE_UNET:
            unet = initialize_unet()
            summary(unet)
            with section("to save unet"):
                unet.save()

    else:

        model = initialize_latent_diffusion(path=CHECKPOINT_PATH, force_submodels_init=True)
        summary(model)
        if GRANULARITY == 0:
            with section("to save vae submodels"):
                model.autoencoder.save_submodels() # saves autoencoder submodels (encoder, decoder) with loaded state dict
            with section("to unload vae submodels"):
                model.autoencoder.unload_submodels() # unloads autoencoder submodels
            with section("to save embedder submodels"):
                model.clip_embedder.save_submodels() # saves text embedder submodels (tokenizer, transformer) with loaded state dict
            with section("to unload embedder submodels"):
                model.clip_embedder.unload_submodels() # unloads text embedder submodels
            with section("to save latent diffusion submodels"):
                model.save_submodels() # saves latent diffusion submodels (autoencoder, clip_embedder) with loaded state dict and unloaded submodels
            with section("to unload latent diffusion submodels"):
                model.unload_submodels() # unloads latent diffusion submodels
            with section("to save latent diffusion model"):
                model.save() # saves latent diffusion model with loaded state dict and unloaded submodels
        elif GRANULARITY == 1:
            with section("to save latent diffusion submodels"):
                model.save_submodels() # saves latent diffusion submodels (autoencoder, clip_embedder and unet) with loaded state dict loaded submodels
            with section("to unload latent diffusion submodels"):
                model.unload_submodels() # unloads latent diffusion submodels
            with section("to save latent diffusion model"):
                model.save() # saves latent diffusion model with loaded state dict and unloaded submodels
        elif GRANULARITY == 2:
            with section("to save latent diffusion model"):
                model.save() # saves latent diffusion model with loaded state dict and loaded submodels