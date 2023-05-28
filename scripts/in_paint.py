"""
---
title: In-paint images using stable diffusion with a prompt
summary: >
 In-paint images using stable diffusion with a prompt
---

# In-paint images using [stable diffusion](../index.html) with a prompt
"""

from typing import Optional

import torch
from labml import monit

from stable_diffusion_base_script import StableDiffusionBaseScript
from stable_diffusion.utils.model import save_images, set_seed, get_autocast
from cli_builder import CLI

def get_model_path():
    return "./input/model/sd-v1-4.ckpt"  

class InPaint(StableDiffusionBaseScript):
    """
    ### Image in-painting class
    """

    @torch.no_grad()
    def repaint_image(self, *,
                 orig_img: str,
                 strength: float,
                 batch_size: int = 3,
                 prompt: str,
                 uncond_scale: float = 5.0,
                 mask: Optional[torch.Tensor] = None,
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

        orig = self.encode_image(orig_img, batch_size)
        mask = self.prepare_mask(mask, orig)
        
        # Noise diffuse the original image
        orig_noise = torch.randn(orig.shape, device=self.device)

        # Get the number of steps to diffuse the original
        t_index = self.calc_strength_time_step(strength)

        # AMP auto casting
        autocast = get_autocast()
        with autocast:
            un_cond, cond = self.get_text_conditioning(uncond_scale, prompts, batch_size)
            
            x = self.paint(orig, cond, t_index, uncond_scale,
                      un_cond, mask, orig_noise)
            # Decode the image from the [autoencoder](../model/autoencoder.html)
            return self.decode_image(x)


def main():
    opt = CLI('In-paint images using stable diffusion with a prompt') \
        .prompt() \
        .checkpoint_path() \
        .orig_img() \
        .output() \
        .batch_size() \
        .steps() \
        .scale() \
        .strength() \
        .force_cpu() \
        .cuda_device() \
        .parse()

    
    set_seed(42)
    
    if opt.strength < 0. or opt.strength > 1.:
        print("ERROR: can only work with strength in [0.0, 1.0]")
        exit(1)


    in_paint = InPaint(checkpoint_path=opt.checkpoint_path,
                       ddim_steps=opt.steps,
                       force_cpu=opt.force_cpu,
                       cuda_device=opt.cuda_device)
    in_paint.initialize_script()

    with monit.section('Generate'):
        images = in_paint.repaint_image(
            orig_img=opt.orig_img,
            strength=opt.strength,
            batch_size=opt.batch_size,
            prompt=opt.prompt,
            uncond_scale=opt.scale
        )

        save_images(images, opt.output)


#
if __name__ == "__main__":
    main()
