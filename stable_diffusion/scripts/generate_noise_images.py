import os, sys
sys.path.append("../../")
import random
from text_to_image import Txt2Img
import torch

# Function to generate a prompt
def generate_prompt(prompt_prefix, artist):
    # Generate the prompt
    prompt = f"{prompt_prefix} {artist}"
    return prompt

# Function to save noise seeds to a file
def save_noise_seeds(num_seeds, output_file):
    noise_seeds = [random.randint(0, 9999) for _ in range(num_seeds)]
    with open(output_file, 'w') as f:
        for seed in noise_seeds:
            f.write(f"{seed}\n")

# Generate prompt
prompt_prefix = "A woman with flowers in her hair in a courtyard, in the style of"
artist_file = os.path.join(os.path.dirname(__file__), '../artists.txt')

# Set the output directory
output_dir = os.path.join(os.path.dirname(__file__), '../../output/noise-tests/')

# Check if CUDA is available
if not torch.cuda.is_available():
    print("WARNING: You are running this script without CUDA. Brace yourself for a slow ride.")

# Load the Stable Diffusion model
checkpoint_path = os.path.join(os.path.dirname(__file__), '../../input/model/sd-v1-4.ckpt')
sampler_name = 'ddim'
txt2img = Txt2Img(checkpoint_path=checkpoint_path, sampler_name=sampler_name, n_steps=20)

# Generate images for each artist
with open(artist_file, 'r') as f:
    artists = f.readlines()
    for artist in artists:
        artist = artist.strip()
        prompt = generate_prompt(prompt_prefix, artist)

        # Generate images for each noise seed
        num_seeds = 8
        noise_file = "noise-seeds.txt"
        save_noise_seeds(num_seeds, noise_file)

        with torch.no_grad():
            for i in range(num_seeds):
                # Use seed
                with open(noise_file, 'r') as f:
                    noise_seed = int(f.readlines()[i].strip())

                # Generate image
                image_name = f"a{i:04d}_n{noise_seed}.jpg"
                dest_path = os.path.join(output_dir, image_name)

                # Check if the image already exists
                if not os.path.exists(dest_path):
                    txt2img(dest_path=dest_path, batch_size=1, prompt=prompt, seed=noise_seed)

# Unload the Stable Diffusion model
del txt2img

print("Images generated successfully!")
