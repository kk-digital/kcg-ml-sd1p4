import os
import sys
import time
import torch

base_dir = os.getcwd()
sys.path.insert(0, base_dir)

from scripts.stable_diffusion_base_script import StableDiffusionBaseScript
from pathlib import Path
from stable_diffusion.utils_image import save_images
    

def test_un_cond_embedding():
    start_time = time.time()

    # initialize options
    output_path = Path("./output/uncond_test")
    cfg_strength = 7
    image_width = 512
    image_height = 512
    batch_size = 1
    positive_prompt = "a painting of a virus monster playing guitar"
    negative_prompt = ""

    # initialize sd model
    sampler = "ddim"
    force_cpu = False
    cuda_device = "cuda:0"
    steps = 20
    checkpoint_path = "./input/model/sd/v1-5-pruned-emaonly/v1-5-pruned-emaonly.safetensors"

    txt2img = StableDiffusionBaseScript(
        sampler_name=sampler,
        n_steps=steps,
        force_cpu=force_cpu,
        cuda_device=cuda_device,
    )
    txt2img.initialize_latent_diffusion(autoencoder=None, clip_text_embedder=None, unet_model=None,
                                        path=checkpoint_path, force_submodels_init=True)


    
    # Get both positive and negative (unconditioned) embeddings
    un_cond, cond = txt2img.get_text_conditioning(cfg_strength, positive_prompt, negative_prompt, batch_size)

    seeds = [i for i in range(100)]  # Keeping this to have 100 iterations
    output_path.mkdir(parents=True, exist_ok=True)
    
    for idx, seed in enumerate(seeds):
        this_seed = seed

        # Iterate over two null_prompt conditions: un_cond and None
        for null_prompt, un_cond_name in zip([un_cond, None], ['""', "None"]):
            (output_path).mkdir(parents=True, exist_ok=True)
            start_time = time.time()

            # Generate the latent representations using the current null_prompt condition
            latent = txt2img.generate_images_latent_from_embeddings(
                        batch_size=batch_size,
                        embedded_prompt=cond,  # Use the positive embedding as the embedded prompt
                        null_prompt=null_prompt,  # Use the current null_prompt condition
                        uncond_scale=cfg_strength,
                        seed=this_seed,
                        w=image_width,
                        h=image_height
                    )
            
            # Get the images from the latent representations
            images = txt2img.get_image_from_latent(latent)
            del latent
            elapsed_time = time.time() - start_time

            image_path = output_path / f"seed_{seed}_{un_cond_name}.jpg"
            _, _ = save_images(images, image_path)

            print(f"Null Prompt Condition: {un_cond_name}, Seed: {seed}, Time taken: {elapsed_time} seconds")

    print(f"Total time for all generations: {time.time() - start_time} seconds")






