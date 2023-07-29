"""
---
title: Utility functions for stable diffusion
summary: >
 Utility functions for stable diffusion
---

# Utility functions for [stable diffusion](index.html)
"""
import os.path
import random
from pathlib import Path
from typing import BinaryIO, List, Optional, Union

import torchvision

import PIL
import numpy as np
import torch
from PIL import Image

from utility.labml import monit
from utility.labml.logger import inspect
from stable_diffusion_reference.latent_diffusion import LatentDiffusion
from stable_diffusion_reference.model.autoencoder import Encoder, Decoder, Autoencoder
from stable_diffusion_reference.model.clip_embedder import CLIPTextEmbedder
from stable_diffusion_reference.model.unet import UNetModel


def set_seed(seed: int):
    """
    ### Set random seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(path: Union[str, Path] = '', device=None) -> LatentDiffusion:
    """
    ### Load [`LatentDiffusion` model](latent_diffusion.html)
    """

    # Initialize the autoencoder

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

    # Initialize the CLIP text embedder

    with monit.section('Initialize CLIP Embedder'):
        clip_text_embedder = CLIPTextEmbedder(
            device=device,
        )

    # Initialize the U-Net

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


def save_image(images: torch.Tensor, dest_path: str, img_format: str = 'jpeg'):
    """
    ### Save an image

    :param images: is the tensor with image of shape `[batch_size, channels, height, width]`
    :param dest_path: is the path and filename to save images in
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


def save_images(images: torch.Tensor, dest_path: str, img_format: str = 'jpeg'):
    """
    ### Save images

    :param images: is the tensor with images of shape `[batch_size, channels, height, width]`
    :param dest_path: is the folder to save images in
    :param img_format: is the image format
    """
    if not os.path.exists(dest_path):
        # If it doesn't exist, create it
        os.makedirs(dest_path)

    # Map images to `[0, 1]` space and clip
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    # Transpose to `[batch_size, height, width, channels]` and convert to numpy
    images = images.cpu()
    images = images.permute(0, 2, 3, 1)
    images = images.float().numpy()

    # Save images
    for i, img in enumerate(images):
        filename = "{0}".format(i).zfill(5)
        filename = "{0}.jpeg".format(filename)
        final_path = os.path.join(dest_path, filename)
        img = Image.fromarray((255. * img).astype(np.uint8))

        img.save(final_path, format=img_format)


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


def get_device(device=None, cuda_fallback: str = 'cuda:0'):
    """
    ### Get device
    """
    if device is None:
        device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
        print(f'INFO: `device` is None. Falling back to current device: {device}.')
    else:
        device = torch.device(device)
    try:
        print(f'INFO: Using CUDA device {device.index}: {torch.cuda.get_device_name(device)}.')
    except Exception as e:
        print(e)
        print(f'Using {device}. Slow on CPU.')

    return device


def get_autocast(force_cpu: bool = False):
    """
    ### Get autocast
    """
    if torch.cuda.is_available() and not force_cpu:
        return torch.cuda.amp.autocast()

    print("WARNING: You are running this script without CUDA. It might be extremely slow.")
    return torch.cpu.amp.autocast()
