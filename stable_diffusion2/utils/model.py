"""
---
title: Utility functions for stable diffusion
summary: >
 Utility functions for stable diffusion
---

# Utility functions for [stable diffusion](index.html)
"""

import random
from pathlib import Path
from typing import Union, BinaryIO, List, Optional


import PIL
import numpy as np
import torch
import torchvision

from PIL import Image

from stable_diffusion2.constants import AUTOENCODER_PATH, ENCODER_PATH, DECODER_PATH
from stable_diffusion2.constants import EMBEDDER_PATH, TOKENIZER_PATH, TRANSFORMER_PATH
from stable_diffusion2.constants import UNET_PATH

from stable_diffusion2.utils.utils import SectionManager as section

from stable_diffusion2.latent_diffusion import LatentDiffusion
from stable_diffusion2.model.vae.encoder import Encoder
from stable_diffusion2.model.vae.decoder import Decoder
from stable_diffusion2.model.vae.autoencoder import Autoencoder
from stable_diffusion2.model.clip.clip_embedder import CLIPTextEmbedder
from stable_diffusion2.model.unet.unet import UNetModel

from transformers import CLIPTokenizer, CLIPTextModel

# from stable_diffusion2.model.unet import UNetModel

def check_device(device, cuda_fallback = 'cuda:0'):
    if device is None:
        device = torch.device(cuda_fallback if torch.cuda.is_available() else 'cpu')
        print(f'Using {device}: {torch.cuda.get_device_name(device)}. Slow on CPU.')
    else:
        device = torch.device(device)
        print(f'Using {device}: {torch.cuda.get_device_name(device)}. Slow on CPU.')
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
    
    device = check_device(device)
    with section('decoder initialization'):
        decoder = Decoder(out_channels=out_channels,
                        z_channels=z_channels,
                        channels=channels,
                        channel_multipliers=channel_multipliers,
                        n_resnet_blocks=n_resnet_blocks).to(device)    
    return decoder
    # Initialize the autoencoder    

    
def initialize_autoencoder(device = None, encoder = None, decoder = None, emb_channels = 4, z_channels = 4, force_submodels_init = True) -> Autoencoder:
    device = check_device(device)
    # Initialize the autoencoder
    with section('autoencoder initialization'):
        if force_submodels_init:
            if encoder is None:
                encoder = initialize_encoder(device=device, z_channels=z_channels)
            if decoder is None:
                decoder = initialize_decoder(device=device, z_channels=z_channels)
        
        autoencoder = Autoencoder(emb_channels=emb_channels,
                                    encoder=encoder,
                                    decoder=decoder,
                                    z_channels=z_channels).to(device)
    return autoencoder
def initialize_tokenizer(device = None, version="openai/clip-vit-large-patch14") -> CLIPTokenizer:
    check_device(device)
    tokenizer = CLIPTokenizer.from_pretrained(version)
    return tokenizer
def initialize_transformer(device = None, version = "openai/clip-vit-large-patch14") -> CLIPTextModel:
    check_device(device)
    transformer = CLIPTextModel.from_pretrained(version).eval().to(device)        
    return transformer
def initialize_clip_embedder(device = None, tokenizer = None, transformer = None, force_submodels_init = True) -> CLIPTextEmbedder:

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

def initialize_latent_diffusion(path: Union[str, Path] = '', device = None, autoencoder = None, clip_text_embedder = None, unet_model = None, force_submodels_init = False) -> LatentDiffusion:
    """
    ### Load [`LatentDiffusion` model](latent_diffusion.html)
    """
    device = check_device(device)
    # Initialize the submodels, if not given
    if force_submodels_init:
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
    with section(f"stable-diffusion checkpoint loading, from {path}"):
        checkpoint = torch.load(path, map_location="cpu")

    # Set model state
    with section('model state loading'):
        missing_keys, extra_keys = model.load_state_dict(checkpoint["state_dict"], strict=False)

    # Debugging output
    # inspect(global_step=checkpoint.get('global_step', -1), missing_keys=missing_keys, extra_keys=extra_keys,
    #         _expand=True)

    #
    model.eval()
    
    return model


def load_img(path: str):
    """
    ### Load an image

    This loads an image from a file and returns a PyTorch tensor.

    :param path: is the path of the image
    """
    # Open Image
    image = Image.open(path).convert("RGB")
    # Get image size
    w, h = image.size
    # Resize to a multiple of 32
    w = w - w % 32
    h = h - h % 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    # Convert to numpy and map to `[-1, 1]` for `[0, 255]`
    image = np.array(image).astype(np.float32) * (2. / 255.0) - 1
    # Transpose to shape `[batch_size, channels, height, width]`
    image = image[None].transpose(0, 3, 1, 2)
    # Convert to torch
    return torch.from_numpy(image)


def save_images(images: torch.Tensor, dest_path: str, img_format: str = 'jpeg'):
    """
    ### Save a images

    :param images: is the tensor with images of shape `[batch_size, channels, height, width]`
    :param dest_path: is the folder to save images in
    :param img_format: is the image format
    """

    # Map images to `[0, 1]` space and clip
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    # Transpose to `[batch_size, height, width, channels]` and convert to numpy
    images = images.cpu()
    images = images.permute(0, 2, 3, 1)
    images = images.float().numpy()

    # Save images
    for i, img in enumerate(images):
        img = Image.fromarray((255. * img).astype(np.uint8))
        img.save(dest_path, format=img_format)

def save_image_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[str, Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = torchvision.utils.make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)        

def get_device(force_cpu: bool = False, cuda_fallback: str = 'cuda:0'):
    """
    ### Get device
    """
    if torch.cuda.is_available() and not force_cpu:
        device_name = torch.cuda.get_device_name(0)
        print("INFO: Using CUDA device: {}".format(device_name))
        return cuda_fallback

    print("WARNING: You are running this script without CUDA. It may be very slow.")
    return 'cpu'

def get_autocast(force_cpu: bool = False):
    """
    ### Get autocast
    """
    if torch.cuda.is_available() and not force_cpu:
        return torch.cuda.amp.autocast()

    print("WARNING: You are running this script without CUDA. It may be very slow.")
    return torch.cpu.amp.autocast()


def set_seed(seed: int):
    """
    ### Set random seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)