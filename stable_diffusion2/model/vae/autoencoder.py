"""
---
title: Autoencoder for Stable Diffusion
summary: >
 Annotated PyTorch implementation/tutorial of the autoencoder
 for stable diffusion.
---

# Autoencoder for [Stable Diffusion](../index.html)

This implements the auto-encoder model used to map between image space and latent space.

We have kept to the model definition and naming unchanged from
[CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
so that we can load the checkpoints directly.
"""

from typing import List

import torch
import os
import torch.nn.functional as F
from torch import nn
from .auxiliary_classes import *
from .encoder import Encoder
from .decoder import Decoder
import os
import sys
sys.path.insert(0, os.getcwd())

from stable_diffusion2.constants import ENCODER_PATH, DECODER_PATH, AUTOENCODER_PATH
from stable_diffusion2.utils.utils import check_device
# ENCODER_PATH = os.path.abspath('./input/model/autoencoder/encoder.ckpt')
# DECODER_PATH = os.path.abspath('./input/model/autoencoder/decoder.ckpt')
# AUTOENCODER_PATH = os.path.abspath('./input/model/autoencoder/autoencoder.ckpt')

class Autoencoder(nn.Module):
    """
    ## Autoencoder

    This consists of the encoder and decoder modules.
    """

    def __init__(self, emb_channels: int, z_channels: int, encoder = None, decoder = None, device = None):
        """
        :param encoder: is the encoder
        :param decoder: is the decoder
        :param emb_channels: is the number of dimensions in the quantized embedding space
        :param z_channels: is the number of channels in the embedding space
        """
        super().__init__()

        self.device = check_device(device)

        self.encoder = encoder
        self.decoder = decoder
        self.z_channels = z_channels
        self.emb_channels = emb_channels
        # Convolution to map from embedding space to
        # quantized embedding space moments (mean and log variance)
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * emb_channels, 1)
        # Convolution to map from quantized embedding space back to
        # embedding space
        self.post_quant_conv = nn.Conv2d(emb_channels, z_channels, 1)
        self.to(self.device)

    def save_submodels(self, encoder_path = ENCODER_PATH, decoder_path = DECODER_PATH):
        """
        ### Save the model to a checkpoint
        """
        self.encoder.save(encoder_path)
        print(f"Encoder saved to: {encoder_path}")
        self.decoder.save(decoder_path)
        print(f"Decoder saved to: {decoder_path}")
        # torch.save(self.encoder, encoder_path)
        # torch.save(self.decoder, decoder_path)

    def save(self, autoencoder_path = AUTOENCODER_PATH):
        """
        ### Save the model to a checkpoint
        """

        torch.save(self, autoencoder_path)
        print(f"Autoencoder saved to: {autoencoder_path}")

    def load_submodels(self, encoder_path = ENCODER_PATH, decoder_path = DECODER_PATH):
        
        """
        ### Load the model from a checkpoint
        """
        self.encoder = torch.load(encoder_path, map_location=self.device)
        self.encoder.eval()
        print(f"Encoder loaded from: {encoder_path}")
        self.decoder = torch.load(decoder_path, map_location=self.device)
        self.decoder.eval()
        print(f"Decoder loaded from: {decoder_path}")
        return self

    def load_encoder(self, encoder_path = ENCODER_PATH):
        self.encoder = torch.load(encoder_path, map_location=self.device)
        self.encoder.eval()
        print(f"Encoder loaded from: {encoder_path}")
        return self.encoder
    
    def load_decoder(self, decoder_path = DECODER_PATH):
        self.decoder = torch.load(decoder_path, map_location=self.device)
        self.decoder.eval()
        print(f"Decoder loaded from: {decoder_path}")
        return self.decoder

    def unload_encoder(self):
        self.encoder.to('cpu')
        del self.encoder
        torch.cuda.empty_cache()
        self.encoder = None

    def unload_decoder(self):
        self.decoder.to('cpu')
        del self.decoder
        torch.cuda.empty_cache()
        self.encoder = None

    def unload_submodels(self):
        self.encoder.to('cpu')
        self.decoder.to('cpu')
        del self.encoder
        del self.decoder
        torch.cuda.empty_cache()
        self.encoder = None
        self.decoder = None

    def encode(self, img: torch.Tensor) -> 'GaussianDistribution':
        """
        ### Encode images to latent representation

        :param img: is the image tensor with shape `[batch_size, img_channels, img_height, img_width]`
        """
        # Get embeddings with shape `[batch_size, z_channels * 2, z_height, z_height]`
        z = self.encoder(img)
        # Get the moments in the quantized embedding space
        moments = self.quant_conv(z)
        # Return the distribution
        return GaussianDistribution(moments)

    def decode(self, z: torch.Tensor):
        """
        ### Decode images from latent representation

        :param z: is the latent representation with shape `[batch_size, emb_channels, z_height, z_height]`
        """
        # Map to embedding space from the quantized representation
        z = self.post_quant_conv(z)
        # Decode the image of shape `[batch_size, channels, height, width]`
        return self.decoder(z)

if __name__ == "__main__":
    prompts = ["", "A painting of a computer virus", "A photo of a computer virus"]

    vae = Autoencoder(emb_channels=4, z_channels=4)


    vae.initialize_submodels()
    # embeddings1 = vae(prompts)

    vae.save()  
    vae.save_submodels()  
    vae.unload_submodels()

    vae.load_submodels()

    # embeddings2 = vae(prompts)

    # assert torch.allclose(embeddings1, embeddings2)

    vae_disk = torch.load(AUTOENCODER_PATH, map_location="cuda:0")
    # print(vae)
    # embeddings3 = vae(prompts)
    # assert torch.allclose(embeddings1, embeddings3)
    # assert torch.allclose(embeddings2, embeddings3)