"""
---
title: Generate images using stable diffusion with a prompt from a given image
summary: >
 Generate images using stable diffusion with a prompt from a given image
---

# Generate images using [stable diffusion](../index.html) with a prompt from a given image
"""

import torch
from labml import monit

from stable_diffusion_base_script import StableDiffusionBaseScript
from stable_diffusion.utils.model import save_images, set_seed, get_autocast
from cli_builder import CLI

class Img2Img(StableDiffusionBaseScript):
    """
    ### Image to image class
    """

    @torch.no_grad()
    def transform_image(self, *,
                 orig_img: str,
                 strength: float,
                 batch_size: int = 3,
                 prompt: str,
                 uncond_scale: float = 5.0,
                 ) -> torch.Tensor:
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

        t_index = self.calc_strength_time_step(strength)

        # AMP auto casting
        autocast = get_autocast()
        with autocast:
            print(f'Generating images with prompt: "{prompt}"')

            un_cond, cond = self.get_text_conditioning(uncond_scale, prompts, batch_size)
            
            x = self.paint(orig, cond, t_index, uncond_scale, un_cond)
            
            # Reconstruct from the noisy image
            return self.decode_image(x)

def main():
    """
    ### CLI
    """
    opt = CLI('Modify an image using a prompt') \
        .prompt() \
        .orig_img() \
        .batch_size() \
        .steps() \
        .scale() \
        .strength() \
        .checkpoint_path() \
        .output() \
        .force_cpu() \
        .cuda_device() \
        .parse()
    
    print(opt)

    set_seed(42)

    print("Chegou aqui")
    img2img = Img2Img(checkpoint_path=opt.checkpoint_path,
                      ddim_steps=opt.steps,
                      force_cpu=opt.force_cpu,
                      cuda_device=opt.cuda_device)
    img2img.initialize_script()

    with monit.section('Generating images'):
        images = img2img.transform_image(
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
