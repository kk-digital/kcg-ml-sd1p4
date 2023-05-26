"""
---
title: Generate images using stable diffusion with a prompt
summary: >
 Generate images using stable diffusion with a prompt
---

# Generate images using [stable diffusion](../index.html) with a prompt
"""

import argparse
import os
import time
from pathlib import Path
from typing import Union

import torch

#import parent directory
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from labml import monit
from stable_diffusion.latent_diffusion import LatentDiffusion
from stable_diffusion.util import save_images, set_seed, get_autocast
from stable_diffusion.model.unet_attention import CrossAttention

from stable_diffusion_base_script import StableDiffusionBaseScript

class Txt2Img(StableDiffusionBaseScript):
    """
    ### Text to image class
    """
    model: LatentDiffusion

    @torch.no_grad()
    def __call__(self, *,
                 seed: int = 0,
                 dest_path: str,
                 batch_size: int = 1,
                 prompt: str,
                 h: int = 512, w: int = 512,
                 uncond_scale: float = 7.5,
                 low_vram: bool = False,
                 ):
        """
        :param seed: the seed to use when generating the images
        :param dest_path: is the path to store the generated images
        :param batch_size: is the number of images to generate in a batch
        :param prompt: is the prompt to generate images with
        :param h: is the height of the image
        :param w: is the width of the image
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param low_vram: whether to limit VRAM usage
        """
        # Number of channels in the image
        c = 4
        # Image to latent space resolution reduction
        f = 8

        if seed == 0:
            seed = time.time_ns() % 2**32

        set_seed(seed)
        # Adjust batch size based on VRAM availability
        if low_vram:
            batch_size = 1

        # Make a batch of prompts
        prompts = batch_size * [prompt]

        # AMP auto casting
        autocast = get_autocast()
        with autocast:
            # In unconditional scaling is not $1$ get the embeddings for empty prompts (no conditioning).
            if uncond_scale != 1.0:
                un_cond = self.model.get_text_conditioning(batch_size * [""])
            else:
                un_cond = None
            # Get the prompt embeddings
            cond = self.model.get_text_conditioning(prompts)
            # [Sample in the latent space](../sampler/index.html).
            # `x` will be of shape `[batch_size, c, h / f, w / f]`
            x = self.sampler.sample(cond=cond,
                                    shape=[batch_size, c, h // f, w // f],
                                    uncond_scale=uncond_scale,
                                    uncond_cond=un_cond)
            # Decode the image from the [autoencoder](../model/autoencoder.html)
            images = self.model.autoencoder_decode(x)

        # Save images
        save_images(images, dest_path)


def main():
    """
    ### CLI
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument("--batch_size", type=int, default=4, help="batch size")

    parser.add_argument(
        "--output",
        type=str,
        dest="output_dir",
        default="./outputs",
        help="Output path to store the generated images",
    )

    parser.add_argument(
        '--sampler',
        dest='sampler_name',
        choices=['ddim', 'ddpm'],
        default='ddim',
        help=f'Set the sampler.',
    )

    parser.add_argument(
        '--checkpoint_path',
        dest='checkpoint_path',
        default='./sd-v1-4.ckpt',
        help='Relative path of the checkpoint file (*.ckpt) (defaults to ./sd-v1-4.ckpt)'
    )

    parser.add_argument("--flash", action='store_true', help="whether to use flash attention")

    parser.add_argument("--steps", type=int, default=50, help="number of sampling steps")

    parser.add_argument("--scale", type=float, default=7.5,
                        help="unconditional guidance scale: "
                             "eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")

    parser.add_argument("--low_vram", action='store_true', help="limit VRAM usage")

    parser.add_argument("--force_cpu", action='store_true', help="force CPU usage")

    parser.add_argument("--cuda_device", type=str, default="cuda:0",
                        help="cuda device to use for generation")

    opt = parser.parse_args()


    # Set flash attention
    CrossAttention.use_flash_attention = opt.flash

    # Starts the text2img
    txt2img = Txt2Img(checkpoint_path=opt.checkpoint_path,
                      sampler_name=opt.sampler_name,
                      n_steps=opt.steps,
                        force_cpu=opt.force_cpu,
                        cuda_device=opt.cuda_device)

    with monit.section('Generate'):
        txt2img(dest_path=opt.output_dir,
                batch_size=opt.batch_size,
                prompt=opt.prompt,
                uncond_scale=opt.scale,
                low_vram=opt.low_vram)


if __name__ == "__main__":
    main()
