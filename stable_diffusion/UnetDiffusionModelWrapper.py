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

from .model.vae.autoencoder import Autoencoder
from .model.clip_text_embedder.clip_text_embedder import CLIPTextEmbedder
from .model.unet.unet import UNetModel
from .constants import (
    AUTOENCODER_PATH,
    UNET_PATH,
    LATENT_DIFFUSION_PATH,
    EMBEDDER_PATH,
    ENCODER_PATH,
    DECODER_PATH,
    TOKENIZER_PATH,
    TRANSFORMER_PATH,
)
from .utils.utils import check_device
# from .utils.utils import SectionManager as section
from labml.monit import section

class UnetDiffusionModelWrapper(nn.Module):
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