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

import os
import sys
import safetensors

from utility.utils_logger import logger
from utility.labml.monit import section

sys.path.insert(0, os.getcwd())
from .auxiliary_classes import *
from .encoder import Encoder
from .decoder import Decoder
from stable_diffusion.utils_backend import get_device
from stable_diffusion.model_paths import VAE_ENCODER_PATH, VAE_DECODER_PATH, VAE_PATH


class Autoencoder(nn.Module):
    """
    ## Autoencoder

    This consists of the encoder and decoder modules.
    """

    def __init__(self, emb_channels: int = 4, z_channels: int = 4, encoder=None, decoder=None, device=None):
        """
        :param encoder: is the encoder
        :param decoder: is the decoder
        :param emb_channels: is the number of dimensions in the quantized embedding space
        :param z_channels: is the number of channels in the embedding space
        """
        super().__init__()

        self.device = get_device(device)

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

    def save_submodels(self, encoder_path=VAE_ENCODER_PATH, decoder_path=VAE_DECODER_PATH):

        self.encoder.save(encoder_path)
        self.decoder.save(decoder_path)

    def save(self, autoencoder_path=VAE_PATH):
        """
        ### Save the model to a checkpoint
        """

        try:
            safetensors.torch.save_model(self, autoencoder_path)
            print(f"Autoencoder saved to: {autoencoder_path}")
        except Exception as e:
            print(f"Autoencoder not saved. Error: {e}")

    def load(self, autoencoder_path=VAE_PATH):
        try:
            safetensors.torch.load_model(self, autoencoder_path, strict=True)
            logger.debug(f"Autoencoder loaded from: {autoencoder_path}")
            self.eval()
            return self
        except Exception as e:
            logger.error(f"Autoencoder not loaded. Error: {e}")
            return None

    def load_submodels(self, encoder_path=VAE_ENCODER_PATH, decoder_path=VAE_DECODER_PATH):

        """
        ### Load the model from a checkpoint
        """

        with section("Loading encoder and decoder"):
            self.encoder = Encoder(device=self.device)
            self.encoder.load(encoder_path=encoder_path)
            logger.debug(f"Encoder loaded from: {encoder_path}")
            self.encoder.eval()
            self.decoder = Decoder(device=self.device)
            self.decoder.load(decoder_path=decoder_path)
            logger.debug(f"Decoder loaded from: {decoder_path}")
            self.decoder.eval()
            return self

    def load_encoder(self, encoder_path=VAE_ENCODER_PATH):

        self.encoder = Encoder(device=self.device)
        self.encoder.load(encoder_path=encoder_path)
        logger.debug(f"Encoder loaded from: {encoder_path}")
        self.encoder.eval()
        return self.encoder

    def load_decoder(self, decoder_path=VAE_DECODER_PATH):

        self.decoder = Decoder(device=self.device)
        self.decoder.load(decoder_path=decoder_path)
        logger.debug(f"Decoder loaded from: {decoder_path}")
        self.decoder.eval()
        return self.decoder

    def unload_encoder(self):
        if self.encoder is not None:
            self.encoder.to('cpu')
            del self.encoder
            torch.cuda.empty_cache()
            print('Encoder unloaded')
            self.encoder = None
        else:
            print('Encoder is already unloaded')

    def unload_decoder(self):
        if self.decoder is not None:
            self.decoder.to('cpu')
            del self.decoder
            torch.cuda.empty_cache()
            print('Decoder unloaded')
            self.decoder = None
        else:
            print('Decoder is already unloaded')

    def unload_submodels(self):
        if self.encoder is not None:
            self.encoder.to('cpu')
            del self.encoder
            self.encoder = None
        if self.decoder is not None:
            self.decoder.to('cpu')
            del self.decoder
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

    vae_disk = torch.load(VAE_PATH, map_location="cuda:0")
    # print(vae)
    # embeddings3 = vae(prompts)
    # assert torch.allclose(embeddings1, embeddings3)
    # assert torch.allclose(embeddings2, embeddings3)
