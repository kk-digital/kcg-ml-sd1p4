"""
---
title: Generate images using stable diffusion with a prompt
summary: >
 Generate images using stable diffusion with a prompt
---

# Generate images using [stable diffusion](../index.html) with a prompt
"""

import time
import os
import sys

base_dir = "./"
sys.path.insert(0, base_dir)

from datetime import datetime
from stable_diffusion.stable_diffusion import StableDiffusion
from stable_diffusion.utils_image import save_images
from utility.labml import monit
from stable_diffusion.model.unet.unet_attention import CrossAttention
from cli_builder import CLI

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


def main():
    opt = CLI('Generate images using stable diffusion with a prompt') \
        .prompt() \
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

    #prompts = get_prompts(opt.prompt, opt.prompts_file)
    prompts = [opt.prompt]

    # Split the numbers_string into a list of substrings using the comma as the delimiter
    seed_string_array = opt.seed.split(',')

    # Convert the elements in the list to integers (optional, if needed)
    seed_array = [int(num) for num in seed_string_array]

    if len(seed_array) == 0:
        seed_array = [0]

    # timestamp = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
    # filename = os.path.join(opt.output, f'{timestamp}.jpg')

    # Set flash attention
    CrossAttention.use_flash_attention = opt.flash

    # Starts the text2img
    sd = StableDiffusion(
                      sampler_name=opt.sampler,
                      n_steps=opt.steps,
                      force_cpu=opt.force_cpu,
                      device=opt.cuda_device
                    )
    # txt2img.initialize_latent_diffusion(autoencoder=None, clip_text_embedder=None, unet_model = None, path=opt.checkpoint_path, force_submodels_init=True)
    sd.quick_initialize().load_submodel_tree()
    with monit.section('Generate', total_steps=len(prompts)) as section:
        for prompt in prompts:
            print(f'Generating images for prompt: "{prompt}"')

            for i in range(opt.num_images):
                print("Generating image " + str(i) + " out of " + str(opt.num_images));
                start_time = time.time()
                timestamp = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
                filename = os.path.join(opt.output, f'{timestamp}-{i}.jpg')

                images = sd.generate_images(
                    batch_size=opt.batch_size,
                    prompt=opt.prompt,
                    uncond_scale=opt.cfg_scale,
                    low_vram=opt.low_vram,
                    seed=seed_array[i % len(seed_array)]
                )

                print(images.shape)
                save_images(images, filename)

                # Capture the ending time
                end_time = time.time()

                # Calculate the execution time
                execution_time = end_time - start_time

                print("Execution Time:", execution_time, "seconds")


if __name__ == "__main__":
    main()
