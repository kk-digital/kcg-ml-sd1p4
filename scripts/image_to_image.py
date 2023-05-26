"""
---
title: Generate images using stable diffusion with a prompt from a given image
summary: >
 Generate images using stable diffusion with a prompt from a given image
---

# Generate images using [stable diffusion](../index.html) with a prompt from a given image
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse

import torch

from labml import monit
from stable_diffusion.util import load_img, save_images, set_seed, get_autocast

from stable_diffusion_base_script import StableDiffusionBaseScript


def get_model_path():
    return "./input/model/sd-v1-4.ckpt"  

class Img2Img(StableDiffusionBaseScript):
    """
    ### Image to image class
    """

    @torch.no_grad()
    def __call__(self, *,
                 dest_path: str,
                 orig_img: str,
                 strength: float,
                 batch_size: int = 3,
                 prompt: str,
                 uncond_scale: float = 5.0,
                 ):
        """
        :param dest_path: is the path to store the generated images
        :param orig_img: is the image to transform
        :param strength: specifies how much of the original image should not be preserved
        :param batch_size: is the number of images to generate in a batch
        :param prompt: is the prompt to generate images with
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        """
        # Make a batch of prompts
        prompts = batch_size * [prompt]
        # Load image
        orig_image = load_img(orig_img).to(self.device)
        # Encode the image in the latent space and make `batch_size` copies of it
        orig = self.model.autoencoder_encode(orig_image).repeat(batch_size, 1, 1, 1)

        # Get the number of steps to diffuse the original
        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_index = int(strength * self.ddim_steps)

        # AMP auto casting
        autocast = get_autocast()
        with autocast:
            print(f'Generating images with prompt: "{prompt}"')
            # In unconditional scaling is not $1$ get the embeddings for empty prompts (no conditioning).
            if uncond_scale != 1.0:
                un_cond = self.model.get_text_conditioning(batch_size * [""])
            else:
                un_cond = None
            # Get the prompt embeddings
            cond = self.model.get_text_conditioning(prompts)
            # Add noise to the original image
            x = self.sampler.q_sample(orig, t_index)
            # Reconstruct from the noisy image
            x = self.sampler.paint(x, cond, t_index,
                                   uncond_scale=uncond_scale,
                                   uncond_cond=un_cond)
            # Decode the image from the [autoencoder](../model/autoencoder.html)
            images = self.model.autoencoder_decode(x)

        # Save images
        save_images(images, dest_path, 'img_')


def main():
    """
    ### CLI
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a cute monkey playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--orig_img",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument("--batch_size", type=int, default=4, help="batch size", )
    parser.add_argument("--steps", type=int, default=50, help="number of ddim sampling steps")

    parser.add_argument("--scale", type=float, default=5.0,
                        help="unconditional guidance scale: "
                             "eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")

    parser.add_argument("--strength", type=float, default=0.75,
                        help="strength for noise: "
                             " 1.0 corresponds to full destruction of information in init image")
    
    parser.add_argument("--checkpoint_path", type=str, default=get_model_path(),
                        help="path to the checkpoint")

    parser.add_argument("--output", type=str, default="./outputs",
                        help="path to store the generated images")
    
    parser.add_argument("--force_cpu", action="store_true",
                        help="force the generation to be on CPU")
    
    parser.add_argument("--cuda_device", type=str, default="cuda:0",
                        help="cuda device to use for generation")

    opt = parser.parse_args()
    set_seed(42)

    if not opt.orig_img:
        print("[ERROR] Please provide a path to the input image (--orig_img)")
        return

    img2img = Img2Img(checkpoint_path=opt.checkpoint_path,
                      ddim_steps=opt.steps,
                      force_cpu=opt.force_cpu,
                      cuda_device=opt.cuda_device)

    img2img(
        dest_path=opt.output,
        orig_img=opt.orig_img,
        strength=opt.strength,
        batch_size=opt.batch_size,
        prompt=opt.prompt,
        uncond_scale=opt.scale)


#
if __name__ == "__main__":
    main()
