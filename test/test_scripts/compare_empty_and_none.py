import os
import sys
import time

import torch

base_dir = os.getcwd()
sys.path.insert(0, base_dir)

from os.path import join
import clip

from ga.prompt_generator import generate_prompts
from model.util_clip import UtilClip
from configs.model_config import ModelPathConfig
from stable_diffusion import StableDiffusion, SDconfigs
from scripts.stable_diffusion_base_script import StableDiffusionBaseScript
from stable_diffusion.utils_backend import get_autocast, set_seed
from stable_diffusion.utils_backend import get_device
from stable_diffusion.utils_image import *
from pathlib import Path
import ga




N_STEPS = 20  # 20, 12
CFG_STRENGTH = 9

DEVICE = get_device()
config = ModelPathConfig()


# Load Stable Diffusion
sd = StableDiffusion(device=DEVICE, n_steps=N_STEPS)
sd.quick_initialize().load_autoencoder(config.get_model(SDconfigs.VAE)).load_decoder(config.get_model(SDconfigs.VAE_DECODER))
sd.model.load_unet(config.get_model(SDconfigs.UNET))
    

def main():
    cfg_strength = 7.5
    image_width = 512
    image_height = 512
    batch_size = 1
    this_seed = 0
    this_prompt = "a painting of a virus monster playing guitar"
    negative_prompts = ""
    
    # Get both positive and negative (unconditioned) embeddings
    un_cond, cond = sd.get_text_conditioning(cfg_strength, [this_prompt], negative_prompts, batch_size)

    seeds = [i for i in range(100)]  # Keeping this to have 100 iterations
    output_path = Path("output_images")
    output_path.mkdir(parents=True, exist_ok=True)
    
    for idx, seed in enumerate(seeds):
        this_seed = seed

        # Iterate over two null_prompt conditions: un_cond and None
        for null_prompt, condition_name in zip([un_cond, None], ["un_cond", "None"]):
            (output_path / condition_name).mkdir(parents=True, exist_ok=True)
            start_time = time.time()

            # Generate the latent representations using the current null_prompt condition
            latent = sd.generate_images_latent_from_embeddings(
                        batch_size=batch_size,
                        embedded_prompt=cond,  # Use the positive embedding as the embedded prompt
                        null_prompt=null_prompt,  # Use the current null_prompt condition
                        uncond_scale=cfg_strength,
                        seed=this_seed,
                        w=image_width,
                        h=image_height
                    )
            
            # Get the images from the latent representations
            images = sd.get_image_from_latent(latent)

            elapsed_time = time.time() - start_time

            # Save each image in the batch
            for img_idx, image in enumerate(images):
                image_path = output_path / condition_name / f"{condition_name}_{seed}_{img_idx}.png"
                image.save(image_path)

            print(f"Null Prompt Condition: {condition_name}, Seed: {seed}, Time taken: {elapsed_time} seconds")

    print(f"Total time for all generations: {time.time() - script_start_time} seconds")

if __name__ == "__main__":
    script_start_time = time.time()
    main()





