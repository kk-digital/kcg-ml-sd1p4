#%%


import os
from typing import Callable
from custom_noise.text_to_image_custom import Txt2Img
import torch
import time
import shutil
from tqdm import tqdm

from stable_diffusion.utils.model import save_images
from cli_builder import CLI

noise_seeds = [
    2982,
    4801,
    1995,
    3598,
    987,
    3688,
    8872,
    762
]
#
# CURRENTLY adding kwargs for custom noise function for the samplers
# TODO adapt the script so that it creates a folder for the outputs of each kind of noise
# TODO adapt the script so that it generates images for each kind of noise
# TODO add kwargs for some samplers' parameters

def log_normal(shape, device, mu=0.0, sigma=0.25):
    return torch.exp(mu + sigma*torch.randn(shape, device=device))

def normal(shape, device='cuda:0', mu=0.0, sigma=1.0):
    return torch.normal(mu, sigma, size=shape, device=device)

DISTRIBUTIONS = {
    'Normal': dict(loc=0.0, scale=1.0),
    'Cauchy': dict(loc=0.0, scale=1.0), 
    'Gumbel': dict(loc=1.0, scale=2.0), 
    'Laplace': dict(loc=0.0, scale=1.0), 
    'Uniform': dict(low=0.0, high=1.0)
}

OUTPUT_DIR = os.path.abspath('./output/noise-tests/')

CLEAR_OUTPUT_DIR = True

def get_all_torch_distributions() -> tuple[list[str], list[type]]:
    
    torch_distributions_names = torch.distributions.__all__
    
    torch_distributions = [
        torch.distributions.__dict__[torch_distribution_name] \
            for torch_distribution_name in torch_distributions_names
            ]
    
    return torch_distributions_names, torch_distributions

def get_torch_distribution_from_name(name: str) -> type:
    return torch.distributions.__dict__[name]

def build_noise_samplers(distributions: dict[str, dict[str, float]]) -> dict[str, Callable]:
    noise_samplers = {
        k: lambda shape, device = None: get_torch_distribution_from_name(k)(**v).sample(shape).to(device) \
                       for k, v in distributions.items()
                       }
    return noise_samplers

def create_folder_structure(distributions_dict: dict[str, dict[str, float]], root_dir: str = OUTPUT_DIR) -> None:
    for distribution_name in distributions_dict.keys():
        
        distribution_outputs = os.path.join(root_dir, distribution_name)
        try:
            os.makedirs(distribution_outputs, exist_ok=True)
        except Exception as e:
            print(e)

NOISE_SAMPLERS = build_noise_samplers(DISTRIBUTIONS)

# Function to generate a prompt
def generate_prompt(prompt_prefix, artist):
    # Generate the prompt
    prompt = f"{prompt_prefix} {artist}"
    return prompt

def init_txt2img(checkpoint_path, sampler_name, n_steps):
    txt2img = Txt2Img(checkpoint_path=checkpoint_path, sampler_name=sampler_name, n_steps=n_steps)
    txt2img.initialize_script()
    return txt2img

def get_all_prompts(prompt_prefix, artist_file):
    with open(artist_file, 'r') as f:
        artists = f.readlines()

    num_seeds = len(noise_seeds)
    number_of_artists = len(artists)
    total_images = num_seeds * number_of_artists

    print(f"Artist count: {number_of_artists}")
    print(f"Seed count: {num_seeds}")
    print(f"Total images: {total_images}")
    
    artists = filter(lambda a: a, map(lambda a: a.strip(), artists))
    prompts = map(lambda a: generate_prompt(prompt_prefix, a), artists)

    return total_images, prompts

def show_summary(total_time, partial_time, total_images, output_dir):
    print("[SUMMARY]")
    print("Total time taken: %.2f seconds" % total_time)
    print("Partial time (without initialization): %.2f seconds" % partial_time)
    print("Total images generated: %s" % total_images)
    print("Images/second: %.2f" % (total_images / total_time))
    print("Images/second (without initialization): %.2f" % (total_images / partial_time))

    print("Images generated successfully at", output_dir)

# TODO add the same args and kwargs that `generate_images` have


def generate_images_from_custom_noise(
        distributions: dict[str, tuple[float, float]], 
        output_dir: str = OUTPUT_DIR, 
        clear_output_dir: bool = CLEAR_OUTPUT_DIR,
        prompt_prefix: str="A woman with flowers in her hair in a courtyard, in the style of",
        artist_file: str=os.path.abspath('./input/artists.txt'),
        checkpoint_path: str=os.path.abspath('./input/model/sd-v1-4.ckpt'),
        sampler_name: str='ddim',
        n_steps: int=20,
        batch_size: int=1,                                      
        ):


    # Clear the output directory
    if clear_output_dir:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    # Create the folder structure for the outputs
    create_folder_structure(distributions, output_dir)

    time_before_initialization = time.time()
    
    txt2img = init_txt2img(checkpoint_path, sampler_name, n_steps)

    time_after_initialization = time.time()

    total_images, prompts = get_all_prompts(prompt_prefix, artist_file)
    
    num_prompts_per_distribution = 1

    # Generate the images

    with torch.no_grad():
        with tqdm(total=total_images, desc='Generating images', ) as pbar:
            for distribution_name, params in distributions.items():
                counter = 0
                print(f"Generating images for {distribution_name}")
                noise_fn = lambda shape, device = None: get_torch_distribution_from_name(distribution_name)(**params).sample(shape).to(device)
                for prompt_index, prompt in enumerate(prompts):
                    if counter <= num_prompts_per_distribution:
                        for seed_index, noise_seed in enumerate(noise_seeds):
                            p_bar_description = f"Generating image {seed_index+prompt_index+1} of {total_images} (distribution: {distribution_name}))"
                            pbar.set_description(p_bar_description)
                            image_name = f"n{noise_seed:04d}_a{prompt_index+1:04d}.jpg"
                            dest_path = os.path.join(os.path.join(output_dir, distribution_name), image_name)
                            
                            images = txt2img.generate_images(
                                batch_size=batch_size,
                                prompt=prompt,
                                seed=noise_seed,
                                noise_fn = noise_fn,
                            )
                            save_images(images, dest_path=dest_path)
                            
                            pbar.update(1)
                            # TODO adjust this loop so that it generates a fixed number of images per distribution instead of all images, less uglily
                        counter += 1
                        print("counter: ", counter)
                    else:
                        counter = 0
                        break
    end_time = time.time()

    show_summary(
        total_time=end_time - time_before_initialization,
        partial_time=end_time - time_after_initialization,
        total_images=total_images,
        output_dir=output_dir
    )        
 

def main():
    generate_images_from_custom_noise(DISTRIBUTIONS)

if __name__ == "__main__":
    main()

# %%
