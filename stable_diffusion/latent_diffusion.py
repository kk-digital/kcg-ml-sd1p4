"""
---
title: Latent Diffusion Models
summary: >
 Annotated PyTorch implementation/tutorial of latent diffusion models from paper
 High-Resolution Image Synthesis with Latent Diffusion Models
---

# Latent Diffusion Models

Latent diffusion models use an auto-encoder to map between image space and
latent space. The diffusion model works on the latent space, which makes it
a lot easier to train.
It is based on paper
[High-Resolution Image Synthesis with Latent Diffusion Models](https://papers.labml.ai/paper/2112.10752).

They use a pre-trained auto-encoder and train the diffusion U-Net on the latent
space of the pre-trained auto-encoder.

For a simpler diffusion implementation refer to our [DDPM implementation](../ddpm/index.html).
We use same notations for $\alpha_t$, $\beta_t$ schedules, etc.
"""

from typing import List

import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file
from .model.vae.autoencoder import Autoencoder, Encoder, Decoder
from .model.clip_text_embedder.clip_text_embedder import CLIPTextEmbedder
from .model.unet.unet import UNetModel
from .constants import (
    AUTOENCODER_PATH,
    UNET_PATH,
    LATENT_DIFFUSION_PATH,
    TEXT_EMBEDDER_PATH,
    ENCODER_PATH,
    DECODER_PATH,
    TOKENIZER_PATH,
    TEXT_MODEL_PATH,
)
from .utils_backend import get_device
from utility.labml.monit import section


class UNetWrapper(nn.Module):
    """
    *This is an empty wrapper class around the [U-Net](model/unet.html).
    We keep this to have the same model structure as
    [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
    so that we do not have to map the checkpoint weights explicitly*.
    """

    def __init__(self, diffusion_model: UNetModel):
        super().__init__()
        self.diffusion_model = diffusion_model

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, context: torch.Tensor):
        return self.diffusion_model(x, time_steps, context)


class LatentDiffusion(nn.Module):
    """
    ## Latent diffusion model

    This contains following components:

    * [AutoEncoder](model/autoencoder.html)
    * [U-Net](model/unet.html) with [attention](model/unet_attention.html)
    * [CLIP embeddings generator](model/clip_embedder.html)
    """

    model: UNetWrapper
    autoencoder: Autoencoder
    clip_embedder: CLIPTextEmbedder

    def __init__(
            self,
            latent_scaling_factor: float,
            n_steps: int,
            linear_start: float,
            linear_end: float,
            unet_model: UNetModel = None,
            autoencoder: Autoencoder = None,
            clip_embedder: CLIPTextEmbedder = None,
            device=None,
    ):
        """
        :param unet_model: is the [U-Net](model/unet.html) that predicts noise
         $\epsilon_\text{cond}(x_t, c)$, in latent space
        :param autoencoder: is the [AutoEncoder](model/autoencoder.html)
        :param clip_embedder: is the [CLIP embeddings generator](model/clip_embedder.html)
        :param latent_scaling_factor: is the scaling factor for the latent space. The encodings of
         the autoencoder are scaled by this before feeding into the U-Net.
        :param n_steps: is the number of diffusion steps $T$.
        :param linear_start: is the start of the $\beta$ schedule.
        :param linear_end: is the end of the $\beta$ schedule.
        """
        super().__init__()

        self.device = get_device(device)

        # Wrap the [U-Net](model/unet.html) to keep the same model structure as
        # [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion).
        self.model = UNetWrapper(unet_model)
        # Auto-encoder
        self.first_stage_model = autoencoder
        # [CLIP embeddings generator](model/clip_embedder.html)
        self.cond_stage_model = clip_embedder
        # Latent space scaling factor
        self.latent_scaling_factor = latent_scaling_factor
        # Number of steps $T$
        self.n_steps = n_steps

        # $\beta$ schedule
        beta = (
                torch.linspace(
                    linear_start ** 0.5, linear_end ** 0.5, n_steps, dtype=torch.float64
                )
                ** 2
        )
        self.beta = nn.Parameter(beta.to(torch.float32), requires_grad=False)
        # $\alpha_t = 1 - \beta_t$
        alpha = 1.0 - beta
        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.alpha_bar = nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)
        self.to(self.device)

    @property
    def autoencoder(self):
        return self.first_stage_model

    @property
    def clip_embedder(self):
        return self.cond_stage_model

    @property
    def unet(self):
        return self.model

    def save_submodels(
            self,
            autoencoder_path=AUTOENCODER_PATH,
            embedder_path=TEXT_EMBEDDER_PATH,
            unet_path=UNET_PATH,
            use_safetensors=True
    ):
        if not use_safetensors:
            self.first_stage_model.save(autoencoder_path=autoencoder_path)
            self.cond_stage_model.save(embedder_path=embedder_path)
            self.model.diffusion_model.save(unet_path=unet_path)
        else:
            save_file(self.first_stage_model.state_dict(), autoencoder_path)
            save_file(self.cond_stage_model.state_dict(), embedder_path)
            save_file(self.model.diffusion_model.state_dict(), unet_path)

    def save(self, latent_diffusion_path=LATENT_DIFFUSION_PATH, use_safetensors=True):
        if not use_safetensors:
            torch.save(self, latent_diffusion_path)
        else:
            save_file(self.state_dict(), latent_diffusion_path)

    def load_autoencoder(self, autoencoder_path=AUTOENCODER_PATH, use_safetensors=True):
        if not use_safetensors:
            self.first_stage_model = torch.load(autoencoder_path, map_location=self.device)
            self.first_stage_model.eval()
            print(f"Autoencoder loaded from: {autoencoder_path}")
            return self.first_stage_model
        else:
            self.first_stage_model = initialize_autoencoder(device=self.device)
            self.first_stage_model.load_state_dict(load_file(autoencoder_path))
            self.first_stage_model.eval()
            print(f"Autoencoder loaded from: {autoencoder_path}")
            return self.first_stage_model

    def unload_autoencoder(self):
        del self.first_stage_model
        torch.cuda.empty_cache()
        self.first_stage_model = None

    def load_unet(self, unet_path=UNET_PATH, use_safetensors=True):
        if not use_safetensors:
            unet = torch.load(unet_path, map_location=self.device)
            unet.eval()
            self.model = UNetWrapper(unet)
            return self.model
        else:
            unet = initialize_unet(device=self.device)
            unet.load_state_dict(load_file(unet_path))
            unet.eval()
            self.model = UNetWrapper(unet)
            return self.model

    def unload_unet(self):
        del self.model
        torch.cuda.empty_cache()
        self.model = None

    def load_clip_embedder(self, embedder_path=TEXT_EMBEDDER_PATH):
        self.cond_stage_model = torch.load(embedder_path, map_location=self.device)
        self.cond_stage_model.eval()
        return self.cond_stage_model

    def unload_clip_embedder(self):
        del self.cond_stage_model
        torch.cuda.empty_cache()
        self.cond_stage_model = None

    def load_submodels(
            self,
            autoencoder_path=AUTOENCODER_PATH,
            embedder_path=TEXT_EMBEDDER_PATH,
            unet_path=UNET_PATH,
            use_safetensors=True
    ):
        """
        ### Load the model from a checkpoint
        """
        if not use_safetensors:
            self.first_stage_model = torch.load(autoencoder_path, map_location=self.device)
            self.first_stage_model.eval()
            self.cond_stage_model = torch.load(embedder_path, map_location=self.device)
            self.cond_stage_model.eval()
            self.model = UNetWrapper(
                torch.load(unet_path, map_location=self.device).eval()
            )
            return self
        else:
            self.first_stage_model = initialize_autoencoder(device=self.device, force_submodels_init=True)
            self.first_stage_model.load_state_dict(load_file(autoencoder_path))
            self.first_stage_model.eval()
            # TODO: Resolve the circular dependency between latent_diffusion.py and utils_model.py
            self.cond_stage_model = initialize_clip_embedder(device=self.device, force_submodels_init=True)
            self.cond_stage_model.load_state_dict(load_file(embedder_path))
            self.cond_stage_model.eval()
            unet = initialize_unet(device=self.device).eval()
            unet.load_state_dict(load_file(unet_path))
            self.model = UNetWrapper(unet)
            return self

    def load_submodel_tree(
            self,
            encoder_path=ENCODER_PATH,
            decoder_path=DECODER_PATH,
            autoencoder_path=AUTOENCODER_PATH,
            embedder_path=TEXT_EMBEDDER_PATH,
            tokenizer_path=TOKENIZER_PATH,
            transformer_path=TEXT_MODEL_PATH,
            unet_path=UNET_PATH,
            use_safetensors=True,
    ):
        with section("load submodel tree"):
            if not use_safetensors:
                self.first_stage_model = torch.load(
                    autoencoder_path, map_location=self.device
                )
                self.first_stage_model.eval()
                self.first_stage_model.load_submodels(
                    encoder_path=encoder_path, decoder_path=decoder_path
                )
                self.cond_stage_model = torch.load(embedder_path, map_location=self.device)
                self.cond_stage_model.eval()
                self.cond_stage_model.load_submodels(
                    tokenizer_path=tokenizer_path, transformer_path=transformer_path
                )
                self.model = UNetWrapper(
                    torch.load(unet_path, map_location=self.device).eval()
                )
                return self
            else:
                self.first_stage_model = initialize_autoencoder(device=self.device).load_submodels(
                    use_safetensors=use_safetensors)
                self.first_stage_model.load_state_dict(load_file(autoencoder_path, device=self.device.type))
                self.first_stage_model.eval()

                self.cond_stage_model = CLIPTextEmbedder(device=self.device)
                self.cond_stage_model.load_submodels(
                    tokenizer_path=tokenizer_path, text_model_path=transformer_path
                )
                self.cond_stage_model.load_state_dict(load_file(embedder_path, device=self.device.type))
                self.cond_stage_model.eval()
                unet = initialize_unet(device=self.device).eval()
                unet.load_state_dict(load_file(unet_path, device=self.device.type))
                self.model = UNetWrapper(unet)
                return self

    def unload_submodels(self):
        del self.first_stage_model
        del self.cond_stage_model
        del self.model
        torch.cuda.empty_cache()
        self.first_stage_model = None
        self.cond_stage_model = None
        self.model = None

    def get_text_conditioning(self, prompts: List[str]):
        """
        ### Get [CLIP embeddings](model/clip_embedder.html) for a list of text prompts
        """
        return self.cond_stage_model(prompts)

    def autoencoder_encode(self, image: torch.Tensor):
        """
        ### Get scaled latent space representation of the image

        The encoder output is a distribution.
        We sample from that and multiply by the scaling factor.
        """
        return (
                self.latent_scaling_factor * self.first_stage_model.encode(image).sample()
        )

    def autoencoder_decode(self, z: torch.Tensor):
        """
        ### Get image from the latent representation

        We scale down by the scaling factor and then decode.
        """
        return self.first_stage_model.decode(z / self.latent_scaling_factor)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor):
        """
        ### Predict noise

        Predict noise given the latent representation $x_t$, time step $t$, and the
        conditioning context $c$.

        $$\epsilon_\text{cond}(x_t, c)$$
        """
        return self.model(x, t, context)


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

    with section('autoencoder initialization'):
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
