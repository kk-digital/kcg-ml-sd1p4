"""
---
title: Utility functions for stable diffusion
summary: >
 Utility functions for stable diffusion
---

# Utility functions for [stable diffusion](index.html)
"""

import random
import os
from pathlib import Path
from typing import Union

import PIL
import numpy as np
import torch
from PIL import Image

from labml import monit
from labml.logger import inspect
from stable_diffusion.latent_diffusion import LatentDiffusion
from stable_diffusion.model.autoencoder import Encoder, Decoder, Autoencoder
from stable_diffusion.model.clip_embedder import CLIPTextEmbedder
from stable_diffusion.model.unet import UNetModel


def initialize_autoencoder(device = 'cuda:0', model_path = None) -> Autoencoder:

    # Initialize the autoencoder
    if model_path is None:
        with monit.section('Initialize autoencoder'):
            encoder = Encoder(z_channels=4,
                            in_channels=3,
                            channels=128,
                            channel_multipliers=[1, 2, 4, 4],
                            n_resnet_blocks=2)

            decoder = Decoder(out_channels=3,
                            z_channels=4,
                            channels=128,
                            channel_multipliers=[1, 2, 4, 4],
                            n_resnet_blocks=2)

            autoencoder = Autoencoder(emb_channels=4,
                                    encoder=encoder,
                                    decoder=decoder,
                                    z_channels=4)
            
            torch.save(autoencoder, './input/model/autoencoder.ckpt')
            
            return autoencoder
    else:
        with monit.section('Initialize autoencoder'):
            autoencoder = torch.load(model_path)
            autoencoder.eval()
            return autoencoder
        
def initialize_clip_embedder(device = 'cuda:0', model_path = None) -> CLIPTextEmbedder:

    # Initialize the CLIP text embedder
    if model_path is None:
        with monit.section('Initialize CLIP Embedder'):
            clip_text_embedder = CLIPTextEmbedder(
                device=device,
            )
            torch.save(clip_text_embedder, './input/models/clip_embedder.ckpt')
            return clip_text_embedder
    else:
        with monit.section('Initialize CLIP Embedder'):
            clip_text_embedder = torch.load(model_path)
            clip_text_embedder.eval()
            return clip_text_embedder

def initialize_unet(device = 'cuda:0', model_path = None) -> UNetModel:
    
    # Initialize the U-Net
    if model_path is None:
        with monit.section('Initialize U-Net'):
            unet_model = UNetModel(in_channels=4,
                                out_channels=4,
                                channels=320,
                                attention_levels=[0, 1, 2],
                                n_res_blocks=2,
                                channel_multipliers=[1, 2, 4, 4],
                                n_heads=8,
                                tf_layers=1,
                                d_cond=768)
            torch.save(unet_model, './input/models/unet.ckpt')
            return unet_model
    else:
        with monit.section('Initialize U-Net'):
            unet_model = torch.load(model_path)
            unet_model.eval()
            return unet_model

def load_model(path: Union[str, Path] = '', device = 'cuda:0', autoencoder = None, clip_text_embedder = None, unet_model = None) -> LatentDiffusion:
    """
    ### Load [`LatentDiffusion` model](latent_diffusion.html)
    """

    
    autoencoder = initialize_autoencoder(device=device, model_path=autoencoder)
    clip_text_embedder = initialize_clip_embedder(device=device, model_path=clip_text_embedder)
    unet_model = initialize_unet(device=device, model_path=unet_model)

    # Initialize the Latent Diffusion model
    with monit.section('Initialize Latent Diffusion model'):
        model = LatentDiffusion(linear_start=0.00085,
                                linear_end=0.0120,
                                n_steps=1000,
                                latent_scaling_factor=0.18215,

                                autoencoder=autoencoder,
                                clip_embedder=clip_text_embedder,
                                unet_model=unet_model)

    # Load the checkpoint
    with monit.section(f"Loading model from {path}"):
        checkpoint = torch.load(path, map_location="cpu")

    # Set model state
    with monit.section('Load state'):
        missing_keys, extra_keys = model.load_state_dict(checkpoint["state_dict"], strict=False)

    # Debugging output
    inspect(global_step=checkpoint.get('global_step', -1), missing_keys=missing_keys, extra_keys=extra_keys,
            _expand=True)

    #
    model.eval()
    torch.save(model, './input/models/model.ckpt')
    return model

if __name__ == '__main__':
    
    if int(os.sys.argv[1]) == 1:
        initialize_clip_embedder()
    elif int(os.sys.argv[1]) == 2:
        initialize_autoencoder()
    elif int(os.sys.argv[1]) == 3:
        initialize_unet()
    elif int(os.sys.argv[1]) == 0:
        initialize_clip_embedder()
        initialize_autoencoder()
        initialize_unet()
    else:
        load_model(path='./input/models/v1-5-pruned-emaonly.ckpt', device='cuda:0')