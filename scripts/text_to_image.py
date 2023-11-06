"""
---
title: Generate images using stable diffusion with a prompt
summary: >
 Generate images using stable diffusion with a prompt
---

# Generate images using [stable diffusion](../index.html) with a prompt
"""

import os
import time
from datetime import datetime
from random import randrange
import sys
import torch

base_dir = "./"
sys.path.insert(0, base_dir)
from cli_builder import CLI
from stable_diffusion.model.unet.unet_attention import CrossAttention
from stable_diffusion.utils_backend import set_seed, get_autocast
from stable_diffusion.utils_image import save_images
from stable_diffusion_base_script import StableDiffusionBaseScript
from utility.utils_argument_parsing import get_seed_array_from_string
from utility.labml import monit


def get_prompts(prompt, prompts_file):
    prompts = []
    if prompts_file is not None:
        with open(prompts_file, 'r') as f:
            prompts_from_file = f.readlines()

        prompts.extend(
            filter(lambda x: len(x) > 0, map(lambda x: x.strip(), prompts_from_file))
        )

    if prompt is not None:
        prompts.append(prompt)

    if len(prompts) == 0:
        prompts = ["a painting of a virus monster playing guitar"]

    return prompts


class Txt2Img(StableDiffusionBaseScript):
    """
    ### Text to image class
    """

    @torch.no_grad()
    def generate_images(self, *,
                        seed: int = 0,
                        batch_size: int = 1,
                        prompt: str,
                        negative_prompt: str,
                        h: int = 512, w: int = 512,
                        uncond_scale: float = 7.5,
                        low_vram: bool = False,
                        noise_fn=torch.randn,
                        temperature: float = 1.0,
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
            seed = time.time_ns() % 2 ** 32

        set_seed(seed)
        # Adjust batch size based on VRAM availability
        if low_vram:
            batch_size = 1

        # Make a batch of prompts
        prompts = batch_size * [prompt]

        # AMP auto casting
        autocast = get_autocast()
        with autocast:
            un_cond, cond = self.get_text_conditioning(uncond_scale, prompts, negative_prompt, batch_size)

            # [Sample in the latent space](../sampler/index.html).
            # `x` will be of shape `[batch_size, c, h / f, w / f]`
            x = self.sampler.sample(cond=cond,
                                    shape=[batch_size, c, h // f, w // f],
                                    uncond_scale=uncond_scale,
                                    uncond_cond=un_cond,
                                    noise_fn=noise_fn,
                                    temperature=temperature)

            return self.get_image_from_latent(x)

    @torch.no_grad()
    def generate_images_from_embeddings(self, *,
                                        seed: int = 0,
                                        batch_size: int = 1,
                                        embedded_prompt: torch.Tensor,
                                        null_prompt: torch.Tensor,
                                        h: int = 512, w: int = 512,
                                        uncond_scale: float = 7.5,
                                        low_vram: bool = False,
                                        noise_fn=torch.randn,
                                        temperature: float = 1.0,
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
            seed = time.time_ns() % 2 ** 32

        set_seed(seed)
        # Adjust batch size based on VRAM availability
        if low_vram:
            batch_size = 1

        # Make a batch of prompts
        # prompts = batch_size * [embedded_prompt]
        # cond = torch.cat(prompts, dim=1)
        cond = embedded_prompt.unsqueeze(0)
        # print("cond shape: ", cond.shape)
        # print("uncond shape: ", null_prompt.shape)
        # prompt_list = ["a painting of a virus monster playing guitar", "a painting of a computer virus "]
        # AMP auto casting
        autocast = get_autocast()
        with autocast:

            # [Sample in the latent space](../sampler/index.html).
            # `x` will be of shape `[batch_size, c, h / f, w / f]`
            x = self.sampler.sample(cond=cond,
                                    shape=[batch_size, c, h // f, w // f],
                                    uncond_scale=uncond_scale,
                                    uncond_cond=null_prompt,
                                    noise_fn=noise_fn,
                                    temperature=temperature)

            return self.get_image_from_latent(x)


def text_to_image(prompt, negative_prompt, output, sampler, checkpoint_path, flash, steps, cfg_scale, low_vram, force_cpu, cuda_device,
                  num_images, seed):
    prompts = [prompt]

    # Split the numbers_string into a list of substrings using the comma as the delimiter
    seed_string_array = seed.split(',')

    seed_array = get_seed_array_from_string(seed, array_size=num_images)

    # Set flash attention
    CrossAttention.use_flash_attention = flash

    # Starts the text2img
    txt2img = Txt2Img(
        sampler_name=sampler,
        n_steps=steps,
        force_cpu=force_cpu,
        cuda_device=cuda_device
    )
    txt2img.initialize_latent_diffusion(autoencoder=None, clip_text_embedder=None, unet_model=None,
                                        path=checkpoint_path, force_submodels_init=True)

    with monit.section('Generate', total_steps=len(prompts)):
        for prompt in prompts:
            print(f'Generating images for prompt: "{prompt}"')

            for i in range(num_images):
                print("Generating image " + str(i) + " out of " + str(num_images));
                timestamp = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
                filename = os.path.join(output, f'{timestamp}-{i}.jpg')

                images = txt2img.generate_images(
                    batch_size=1,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    uncond_scale=cfg_scale,
                    low_vram=low_vram,
                    seed=seed_array[i % len(seed_array)]
                )

                save_images(images, filename)

def main():
    opt = CLI('Generate images using stable diffusion with a prompt') \
        .prompt() \
        .negative_prompt() \
        .prompts_file(check_exists=True, required=False) \
        .batch_size() \
        .output() \
        .sampler() \
        .checkpoint_path() \
        .flash() \
        .steps() \
        .cfg_scale() \
        .low_vram() \
        .force_cpu() \
        .cuda_device() \
        .num_images() \
        .seed() \
        .parse()

    text_to_image(opt.prompt, opt.negative_prompt, opt.output, opt.sampler, opt.checkpoint_path, opt.flash, opt.steps,
                  opt.cfg_scale, opt.low_vram, opt.force_cpu, opt.cuda_device, opt.num_images, opt.seed)


if __name__ == "__main__":
    main()
