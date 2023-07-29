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

import time
from safetensors.torch import save_file, load_file

from stable_diffusion.utils_backend import get_device
from .auxiliary_classes import *
from .encoder import Encoder
from .decoder import Decoder
import os
import sys

sys.path.insert(0, os.getcwd())

from stable_diffusion.constants import ENCODER_PATH, DECODER_PATH, AUTOENCODER_PATH


class SectionManager:
    def __init__(self, name: str):
        self.t0 = None
        self.t1 = None
        self.section_name = name

    def __enter__(self):
        print(f"Starting section: {self.section_name}...")
        self.t0 = time.time()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.t1 = time.time()
        print(f"Finished section: {self.section_name} in {(self.t1 - self.t0):.2f} seconds\n")


class Autoencoder(nn.Module):
    """
    ## Autoencoder

    This consists of the encoder and decoder modules.
    """

    def __init__(self, emb_channels: int, z_channels: int, encoder=None, decoder=None, device=None):
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

    def save_submodels(self, encoder_path=ENCODER_PATH, decoder_path=DECODER_PATH, use_safetensors=True):
        """
        ### Save the model to a checkpoint
        """
        self.encoder.save(encoder_path, use_safetensors=use_safetensors)
        print(f"Encoder saved to: {encoder_path}")
        self.decoder.save(decoder_path, use_safetensors=use_safetensors)
        print(f"Decoder saved to: {decoder_path}")
        # torch.save(self.encoder, encoder_path)
        # torch.save(self.decoder, decoder_path)

    def save(self, autoencoder_path=AUTOENCODER_PATH, use_safetensors=True):
        """
        ### Save the model to a checkpoint
        """
        if not use_safetensors:
            torch.save(self, autoencoder_path)
            print(f"Autoencoder saved to: {autoencoder_path}")
        else:
            save_file(self.state_dict(), autoencoder_path)
            print(f"Autoencoder saved to: {autoencoder_path}")

    def load_submodels(self, encoder_path=ENCODER_PATH, decoder_path=DECODER_PATH, use_safetensors=True):

        """
        ### Load the model from a checkpoint
        """
        if not use_safetensors:
            self.encoder = torch.load(encoder_path, map_location=self.device)
            self.encoder.eval()
            print(f"Encoder loaded from: {encoder_path}")
            self.decoder = torch.load(decoder_path, map_location=self.device)
            self.decoder.eval()
            print(f"Decoder loaded from: {decoder_path}")
            return self
        else:
            device = "cpu" if self.device.type == "mps" else self.device  # mps doesn't have support for safe tensors
            self.encoder = initialize_encoder(device=self.device)
            self.encoder.load_state_dict(load_file(encoder_path, device=device))
            self.encoder.eval()
            print(f"Encoder loaded from: {encoder_path}")
            self.decoder = initialize_encoder(device=self.device)
            self.decoder.load_state_dict(load_file(decoder_path, device=device))
            self.decoder.eval()
            print(f"Decoder loaded from: {decoder_path}")
            return self

    def load_encoder(self, encoder_path=ENCODER_PATH, use_safetensors=True):
        if not use_safetensors:
            self.encoder = torch.load(encoder_path, map_location=self.device)
            self.encoder.eval()
            print(f"Encoder loaded from: {encoder_path}")
            return self.encoder
        else:
            self.encoder = initialize_encoder(device=self.device)
            self.encoder.load_state_dict(load_file(encoder_path, device=self.device))
            self.encoder.eval()
            print(f"Encoder loaded from: {encoder_path}")
            return self.encoder

    def load_decoder(self, decoder_path=DECODER_PATH, use_safetensors=True):
        if not use_safetensors:
            self.decoder = torch.load(decoder_path, map_location=self.device)
            self.decoder.eval()
            print(f"Decoder loaded from: {decoder_path}")
            return self.decoder
        else:
            self.decoder = initialize_decoder(device=self.device)
            device = "cpu" if self.device.type == "mps" else self.device  # mps doesn't have support for safe tensors
            self.decoder.load_state_dict(load_file(decoder_path, device=device))
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


def initialize_encoder(device=None,
                       z_channels=4,
                       in_channels=3,
                       channels=128,
                       channel_multipliers=[1, 2, 4, 4],
                       n_resnet_blocks=2) -> Encoder:
    with SectionManager('encoder initialization'):
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
    with SectionManager('decoder initialization'):
        device = get_device(device)
        decoder = Decoder(out_channels=out_channels,
                          z_channels=z_channels,
                          channels=channels,
                          channel_multipliers=channel_multipliers,
                          n_resnet_blocks=n_resnet_blocks).to(device)
    return decoder


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
