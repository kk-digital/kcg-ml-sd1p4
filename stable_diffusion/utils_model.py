"""
---
title: Utility functions for stable diffusion
summary: >
 Utility functions for stable diffusion
---

# Utility functions for [stable diffusion](index.html)
"""

from pathlib import Path
from typing import Union

import safetensors
import torch
from transformers import CLIPTokenizer, CLIPTextModel

from stable_diffusion.model_paths import VAE_PATH, VAE_ENCODER_PATH, VAE_DECODER_PATH
from stable_diffusion.model_paths import LATENT_DIFFUSION_PATH
from stable_diffusion.model_paths import CLIP_TEXT_EMBEDDER_PATH, CLIP_TOKENIZER_DIR_PATH, CLIP_TEXT_MODEL_DIR_PATH
from stable_diffusion.model_paths import UNET_PATH
from stable_diffusion.latent_diffusion import LatentDiffusion
from stable_diffusion.model.clip_text_embedder import CLIPTextEmbedder
from stable_diffusion.model.unet import UNetModel
from stable_diffusion.model.vae import Autoencoder, Encoder, Decoder
from stable_diffusion.utils_backend import get_device
from utility.labml.monit import section


def initialize_encoder(device=None,
                       z_channels=4,
                       in_channels=3,
                       channels=128,
                       channel_multipliers=[1, 2, 4, 4],
                       n_resnet_blocks=2) -> Encoder:
    with section('encoder initialization'):
        device = get_device(device)
        # Initialize the encoder
        encoder = Encoder(z_channels=z_channels,
                          in_channels=in_channels,
                          channels=channels,
                          channel_multipliers=channel_multipliers,
                          n_resnet_blocks=n_resnet_blocks).to(device)
    return encoder


def initialize_decoder(device=None,
                       out_channels=3,
                       z_channels=4,
                       channels=128,
                       channel_multipliers=[1, 2, 4, 4],
                       n_resnet_blocks=2) -> Decoder:
    with section('decoder initialization'):
        device = get_device(device)
        decoder = Decoder(out_channels=out_channels,
                          z_channels=z_channels,
                          channels=channels,
                          channel_multipliers=channel_multipliers,
                          n_resnet_blocks=n_resnet_blocks).to(device)
    return decoder
    # Initialize the autoencoder


def initialize_autoencoder(device=None, encoder=None, decoder=None, emb_channels=4, z_channels=4,
                           force_submodels_init=False) -> Autoencoder:
    # Initialize the autoencoder

    with section('Autoencoder initialization'):
        device = get_device(device)
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


def load_autoencoder(path: Union[str, Path] = VAE_PATH, device=None) -> Autoencoder:
    """
    ### Load [`Autoencoder` model](autoencoder.html)
    """

    # Load the checkpoint
    with section(f"autoencoder model loading, from {path}"):
        device = get_device(device)
        autoencoder = torch.load(path, map_location=device).eval()

    # with section(f"casting autoencoder model to device and evaling"):
    #     autoencoder.to(device)
    #     autoencoder.eval()
    return autoencoder


def load_encoder(path: Union[str, Path] = VAE_ENCODER_PATH, device=None) -> Encoder:
    with section(f"encoder model loading, from {path}"):
        device = get_device(device)
        encoder = torch.load(path, map_location=device).eval()

    # with section(f"casting encoder model to device and evaling"):
    #     encoder.to(device)
    #     encoder.eval()
    return encoder


def load_decoder(path: Union[str, Path] = VAE_DECODER_PATH, device=None) -> Decoder:
    with section(f"decoder model loading, from {path}"):
        device = get_device(device)
        decoder = torch.load(path, map_location=device).eval()

    # with section(f"casting decoder model to device and evaling"):
    #     decoder.to(device)
    #     decoder.eval()
    return decoder


def initialize_tokenizer(device=None, version="openai/clip-vit-large-patch14") -> CLIPTokenizer:
    get_device(device)
    tokenizer = CLIPTokenizer.from_pretrained(version)
    return tokenizer


def load_tokenizer(path: Union[str, Path] = CLIP_TOKENIZER_DIR_PATH, device=None) -> CLIPTokenizer:
    with section(f"CLIP tokenizer loading, from {path}"):
        device = get_device(device)
        tokenizer = torch.load(path, map_location=device).eval()

    return tokenizer


def initialize_transformer(device=None, version="openai/clip-vit-large-patch14") -> CLIPTextModel:
    transformer = CLIPTextModel.from_pretrained(version).eval().to(get_device(device))
    return transformer


def load_transformer(path: Union[str, Path] = CLIP_TEXT_MODEL_DIR_PATH, device=None) -> CLIPTextModel:
    with section(f"CLIP transformer loading, from {path}"):
        device = get_device(device)
        transformer = torch.load(path, map_location=device).eval()

    return transformer


def initialize_clip_embedder(device=None, init_transformer=False) -> CLIPTextEmbedder:
    # Initialize the CLIP text embedder

    with section('CLIP Embedder initialization'):
        device = get_device(device)
        clip_text_embedder = CLIPTextEmbedder(
            device=device,
        )

        if init_transformer:
            initialize_transformer().save_pretrained(CLIP_TEXT_MODEL_DIR_PATH)

        # This is temporary, we should call load_submodels instead
        clip_text_embedder.load_submodels_auto()

        clip_text_embedder.to(device)

    return clip_text_embedder


def load_clip_embedder(path: Union[str, Path] = CLIP_TEXT_EMBEDDER_PATH, device=None) -> CLIPTextEmbedder:
    with section(f"CLIP embedder loading, from {path}"):
        device = get_device(device)
        clip_text_embedder = torch.load(path, map_location=device).eval()

    return clip_text_embedder


def initialize_unet(device=None,
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
    device = get_device(device)
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


def load_unet(path: Union[str, Path] = UNET_PATH, device=None) -> UNetModel:
    with section(f"U-Net model loading, from {path}"):
        device = get_device(device)
        unet_model = torch.load(path, map_location=device).eval()

    return unet_model


def initialize_latent_diffusion(path: Union[str, Path] = None, device=None, autoencoder=None, clip_text_embedder=None,
                                unet_model=None, force_submodels_init=False) -> LatentDiffusion:
    """
    ### Load [`LatentDiffusion` model](latent_diffusion.html)
    """
    device = get_device(device)
    # Initialize the submodels, if not given
    if force_submodels_init:
        if autoencoder is None:
            autoencoder = initialize_autoencoder(device=device, force_submodels_init=force_submodels_init)
        if clip_text_embedder is None:
            clip_text_embedder = CLIPTextEmbedder(device=device).init_submodels()
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
    if path is not None:
        # Load the checkpoint

        with section(f"stable diffusion checkpoint loading, from {path}"):
            tensors_dict = safetensors.torch.load_file(path, device="cpu")

        # Set model state
        with section('model state loading'):
            missing_keys, extra_keys = model.load_state_dict(tensors_dict, strict=False)
            print(f"missing keys {len(missing_keys)}: {missing_keys}")
            print(f"extra keys {len(extra_keys)}: {extra_keys}")

        # Debugging output
        # inspect(global_step=checkpoint.get('global_step', -1), missing_keys=missing_keys, extra_keys=extra_keys,
        #         _expand=True)

        #
    model.to(device)

    return model.eval()


def load_latent_diffusion(path: Union[str, Path] = LATENT_DIFFUSION_PATH, device=None) -> LatentDiffusion:
    """
    ### Load [`LatentDiffusion` model](latent_diffusion.html)
    """
    # Load the checkpoint
    with section(f"loading latent diffusion, from {path}"):
        device = get_device(device)
        latent_diffusion_model = torch.load(path, map_location=device).eval()

    # Initialize the Latent Diffusion model

    return latent_diffusion_model
