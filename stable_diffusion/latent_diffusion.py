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

import safetensors
import torch
import torch.nn as nn
from diffusers.models.vae import DiagonalGaussianDistribution

from utility.labml.monit import section
from .model_paths import (
    VAE_PATH,
    UNET_PATH,
    LATENT_DIFFUSION_PATH,
    CLIP_TEXT_EMBEDDER_PATH,
    VAE_ENCODER_PATH,
    VAE_DECODER_PATH,
    CLIP_TOKENIZER_DIR_PATH,
    CLIP_TEXT_MODEL_DIR_PATH,
)
from .model.clip_text_embedder.clip_text_embedder import CLIPTextEmbedder
from .model.unet.unet import UNetModel
from .model.vae.autoencoder import Autoencoder
from .utils_backend import get_device


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
            latent_scaling_factor: float = 0.18215,
            n_steps: int = 1000,
            linear_start: float = 0.00085,
            linear_end: float = 0.0120,
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
            autoencoder_path=VAE_PATH,
            embedder_path=CLIP_TEXT_EMBEDDER_PATH,
            unet_path=UNET_PATH
    ):

        self.first_stage_model.save(autoencoder_path=autoencoder_path)
        self.cond_stage_model.save(embedder_path=embedder_path)
        self.model.diffusion_model.save(unet_path=unet_path)

    def save(self, latent_diffusion_path=LATENT_DIFFUSION_PATH):
        try:
            safetensors.torch.save_model(self, latent_diffusion_path)
            print(f"Latent diffusion model saved at: {latent_diffusion_path}")
        except Exception as e:
            print(f"Failed to save latent diffusion model at: {latent_diffusion_path}. Error: {e}")

    def load(self, latent_diffusion_path=LATENT_DIFFUSION_PATH):
        try:
            safetensors.torch.load_model(self, latent_diffusion_path, strict=False)
            print(f"Latent diffusion model loaded from: {latent_diffusion_path}")
            return self
        except Exception as e:
            print(f"Failed to load latent diffusion model from: {latent_diffusion_path}. Error: {e}")
            return None

    def load_autoencoder(self, autoencoder_path=VAE_PATH):
        try:
            self.first_stage_model = Autoencoder(device=self.device)
            self.first_stage_model.load(autoencoder_path=autoencoder_path)
            return self.first_stage_model
        except Exception as e:
            print(f"Failed to load autoencoder from: {autoencoder_path}. Error: {e}")
            return None

    def unload_autoencoder(self):
        if self.first_stage_model is not None:
            self.first_stage_model.to('cpu')
            del self.first_stage_model
            torch.cuda.empty_cache()
            print("Autoencoder unloaded")
            self.first_stage_model = None
        else:
            print("Autoencoder not loaded")

    def load_unet(self, unet_path=UNET_PATH):
        try:
            unet = UNetModel(device=self.device)
            unet.load(unet_path=unet_path)

            self.model = UNetWrapper(unet)
            return self.model
        except Exception as e:
            print(f"Failed to load UNet from: {unet_path}. Error: {e}")
            return None

    def unload_unet(self):
        if self.model is not None:
            self.model.to('cpu')
            del self.model
            torch.cuda.empty_cache()
            print("UNet unloaded")
            self.model = None
        else:
            print("UNet not loaded")

    def load_clip_embedder(self, embedder_path=CLIP_TEXT_EMBEDDER_PATH):
        self.cond_stage_model = CLIPTextEmbedder(device=self.device)
        self.cond_stage_model.load(embedder_path=embedder_path)
        return self.cond_stage_model

    def unload_clip_embedder(self):
        if self.cond_stage_model is not None:
            self.cond_stage_model.to('cpu')
            del self.cond_stage_model
            torch.cuda.empty_cache()
            print("CLIPTextEmbedder unloaded")
            self.cond_stage_model = None
        else:
            print("CLIPTextEmbedder not loaded")

    def load_submodels(
            self,
            autoencoder_path=VAE_PATH,
            embedder_path=CLIP_TEXT_EMBEDDER_PATH,
            unet_path=UNET_PATH,
    ):
        with section("Autoencoder"):
            self.load_autoencoder(autoencoder_path=autoencoder_path)
        with section("UNet"):
            self.load_unet(unet_path=unet_path)
        with section("CLIPTextEmbedder"):
            self.load_clip_embedder(embedder_path=embedder_path)
        return self

    def load_submodel_tree(
            self,
            encoder_path=VAE_ENCODER_PATH,
            decoder_path=VAE_DECODER_PATH,
            autoencoder_path=VAE_PATH,
            embedder_path=CLIP_TEXT_EMBEDDER_PATH,
            tokenizer_path=CLIP_TOKENIZER_DIR_PATH,
            transformer_path=CLIP_TEXT_MODEL_DIR_PATH,
            unet_path=UNET_PATH,
    ):
        with section("Load submodel tree"):
            self.load_submodels(autoencoder_path=autoencoder_path, embedder_path=embedder_path, unet_path=unet_path)
            self.first_stage_model.load_submodels(encoder_path=encoder_path, decoder_path=decoder_path)
            self.cond_stage_model.load_submodels(tokenizer_path=tokenizer_path, transformer_path=transformer_path)
        return self

    def unload_submodels(self):

        self.unload_autoencoder()
        self.unload_unet()
        self.unload_clip_embedder()

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

    def get_learned_conditioning(self, c):
        if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
            c = self.cond_stage_model.encode(c)
            if isinstance(c, DiagonalGaussianDistribution):
                c = c.mode()
        else:
            c = self.cond_stage_model(c)

        return c

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor):
        """
        ### Predict noise

        Predict noise given the latent representation $x_t$, time step $t$, and the
        conditioning context $c$.

        $$\epsilon_\text{cond}(x_t, c)$$
        """
        return self.model(x, t, context)
