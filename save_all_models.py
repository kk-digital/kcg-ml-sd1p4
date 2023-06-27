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

# from labml import monit
# from labml.logger import inspect


from stable_diffusion2.constants import AUTOENCODER_PATH, ENCODER_PATH, DECODER_PATH
from stable_diffusion2.constants import EMBEDDER_PATH, TOKENIZER_PATH, TRANSFORMER_PATH
from stable_diffusion2.constants import UNET_PATH

from stable_diffusion2.utils.utils import SectionManager as section

from stable_diffusion2.latent_diffusion import LatentDiffusion
from stable_diffusion2.model2.vae.encoder import Encoder
from stable_diffusion2.model2.vae.decoder import Decoder
from stable_diffusion2.model2.vae.autoencoder import Autoencoder
from stable_diffusion2.model2.clip.clip_embedder import CLIPTextEmbedder
from stable_diffusion2.model2.unet.unet import UNetModel
# from stable_diffusion2.model.unet import UNetModel

def check_device(device):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}. Slow on CPU.')
    else:
        device = torch.device(device)
        print(f'Using device: {device}.')
    return device


def initialize_encoder(device = None, 
                        z_channels=4,
                        in_channels=3,
                        channels=128,
                        channel_multipliers=[1, 2, 4, 4],
                        n_resnet_blocks=2) -> Encoder:
    
    device = check_device(device)
    # Initialize the encoder
    with section('encoder initialization'):
        encoder = Encoder(z_channels=z_channels,
                        in_channels=in_channels,
                        channels=channels,
                        channel_multipliers=channel_multipliers,
                        n_resnet_blocks=n_resnet_blocks).to(device)
    return encoder

def initialize_decoder(device = None, 
                        out_channels=3,
                        z_channels=4,
                        channels=128,
                        channel_multipliers=[1, 2, 4, 4],
                        n_resnet_blocks=2) -> Decoder:
    
    check_device(device)
    with section('decoder initialization'):
        decoder = Decoder(out_channels=out_channels,
                        z_channels=z_channels,
                        channels=channels,
                        channel_multipliers=channel_multipliers,
                        n_resnet_blocks=n_resnet_blocks).to(device)    
    return decoder
    # Initialize the autoencoder    

    
def initialize_autoencoder(device = None, encoder = None, decoder = None, emb_channels = 4, z_channels = 4) -> Autoencoder:
    device = check_device(device)
    # Initialize the autoencoder
    with section('autoencoder initialization'):
        if encoder is None:
            encoder = initialize_encoder(device=device, z_channels=z_channels)
        if decoder is None:
            decoder = initialize_decoder(device=device, z_channels=z_channels)
        
        autoencoder = Autoencoder(emb_channels=emb_channels,
                                    encoder=encoder,
                                    decoder=decoder,
                                    z_channels=z_channels).to(device)
    return autoencoder
        
def initialize_clip_embedder(device = None, tokenizer = None, transformer = None) -> CLIPTextEmbedder:

    # Initialize the CLIP text embedder
    device = check_device(device)
    with section('CLIP Embedder initialization'):
        clip_text_embedder = CLIPTextEmbedder(
                device=device,
            )
        if tokenizer is None:
            clip_text_embedder.load_tokenizer_from_lib()
        else:
            clip_text_embedder.tokenizer = tokenizer
        if transformer is None:
            clip_text_embedder.load_transformer_from_lib()
        else:
            clip_text_embedder.transformer = transformer
        
        clip_text_embedder.to(device)

    return clip_text_embedder

def initialize_unet(device = None, 
                    in_channels=4,
                    out_channels=4,
                    channels=320,
                    attention_levels=[0, 1, 2],
                    n_res_blocks=2,
                    channel_multipliers=[1, 2, 4, 4],
                    n_heads=8,
                    tf_layers=1,
                    d_cond=768) -> UNetModel:

    # Initialize the U-Net
    device = check_device(device)
    with section('U-Net initialization'):
        unet_model = UNetModel(in_channels=in_channels,
                                out_channels=out_channels,
                                channels=channels,
                                attention_levels=attention_levels,
                                n_res_blocks=n_res_blocks,
                                channel_multipliers=channel_multipliers,
                                n_heads=n_heads,
                                tf_layers=tf_layers,
                                d_cond=d_cond).to(device)
            # unet_model.save()
            # torch.save(unet_model, UNET_PATH)
    return unet_model

def load_model(path: Union[str, Path] = '', device = None, autoencoder = None, clip_text_embedder = None, unet_model = None) -> LatentDiffusion:
    """
    ### Load [`LatentDiffusion` model](latent_diffusion.html)
    """
    device = check_device(device)
    # Initialize the models, if not given
    if autoencoder is None:
        autoencoder = initialize_autoencoder(device=device)
    if clip_text_embedder is None:
        clip_text_embedder = initialize_clip_embedder(device=device)
    if unet_model is None:
        unet_model = initialize_unet(device=device)

    # Initialize the Latent Diffusion model
    with section('Latent Diffusion model initialization'):
        model = LatentDiffusion(linear_start=0.00085,
                                linear_end=0.0120,
                                n_steps=1000,
                                latent_scaling_factor=0.18215,

                                autoencoder=autoencoder,
                                clip_embedder=clip_text_embedder,
                                unet_model=unet_model)

    # Load the checkpoint
    with section(f"Loading stable-diffusion checkpoint from {path}"):
        checkpoint = torch.load(path, map_location="cpu")

    # Set model state
    with section('Loading model state'):
        missing_keys, extra_keys = model.load_state_dict(checkpoint["state_dict"], strict=False)

    # Debugging output
    # inspect(global_step=checkpoint.get('global_step', -1), missing_keys=missing_keys, extra_keys=extra_keys,
    #         _expand=True)

    #
    model.eval()
    
    return model

if __name__ == '__main__':
    assert len(os.sys.argv) > 1, 'Please provide an argument.'
    if int(os.sys.argv[1]) == 1:
        embedder = initialize_clip_embedder()
        embedder.save_submodels()
        embedder.save()
    elif int(os.sys.argv[1]) == 2:
        autoencoder = initialize_autoencoder()
        autoencoder.save_submodels()
        autoencoder.save()
    elif int(os.sys.argv[1]) == 3:
        unet = initialize_unet()
        unet.save()
    elif int(os.sys.argv[1]) == 0:
        
        embedder = initialize_clip_embedder()
        embedder.save_submodels()
        embedder.save()
        
        autoencoder = initialize_autoencoder()
        autoencoder.save_submodels()
        autoencoder.save()
        
        unet = initialize_unet()
        unet.save()
    
    else:
        model = load_model(path='./input/model/v1-5-pruned-emaonly.ckpt')
        print(type(model))