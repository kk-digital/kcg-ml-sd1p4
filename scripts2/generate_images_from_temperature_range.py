import os
import sys
import torch
import shutil
import time
import argparse

from typing import Callable
from tqdm import tqdm

# from stable_diffusion2.utils.model import save_images, save_image_grid
from auxiliary_functions import save_images, save_image_grid, get_torch_distribution_from_name

from text_to_image import Txt2Img

from stable_diffusion2.latent_diffusion import LatentDiffusion
from stable_diffusion2.constants import CHECKPOINT_PATH, AUTOENCODER_PATH, UNET_PATH, EMBEDDER_PATH, LATENT_DIFFUSION_PATH, ENCODER_PATH, DECODER_PATH, TOKENIZER_PATH, TRANSFORMER_PATH
from stable_diffusion2.utils.utils import SectionManager as section
from stable_diffusion2.utils.model import initialize_latent_diffusion, initialize_autoencoder, initialize_clip_embedder
import safetensors.torch as st

# CHECKPOINT_PATH = os.path.abspath('./input/model/v1-5-pruned-emaonly.ckpt')




NOISE_SEEDS = [
    2982,
    4801,
    1995,
    3598,
    987,
    3688,
    8872,
    762
]

NUM_SEEDS = 1
NOISE_SEEDS = NOISE_SEEDS[:NUM_SEEDS]
NUM_ARTISTS = 1
OUTPUT_DIR = os.path.abspath('./output/noise-tests/')

parser = argparse.ArgumentParser(
        description='')
parser.add_argument('-fd', '--from_disk', type=bool, default=True)
parser.add_argument('-d', '--distribution', type=int, default=4)
parser.add_argument('--num_distributions', type=int, default=3)
parser.add_argument('--params_range', nargs = "+", type=float, default=[0.49, 0.54])
# parser.add_argument('-t', '--temperature', type=float, default=1.0)
parser.add_argument('--temperature_steps', type=int, default=5)
parser.add_argument('--temperature_range', nargs = "+", type=float, default=[1.0, 4.0])
parser.add_argument('--ddim_eta', type=float, default=0.1)
parser.add_argument('--latent_diffusion_init_mode', type=int, default=0)
parser.add_argument('--txt2img_init_from_saved', type=bool, default=False)
args = parser.parse_args() 
FROM_DISK = args.from_disk

DISTRIBUTION = args.distribution
NUM_DISTRIBUTIONS = args.num_distributions
PARAMS_RANGE = args.params_range
TEMPERATURE_STEPS = args.temperature_steps
TEMPERATURE_RANGE = args.temperature_range
DDIM_ETA = args.ddim_eta

TEMP_RANGE = torch.linspace(*TEMPERATURE_RANGE, TEMPERATURE_STEPS)

CLEAR_OUTPUT_DIR = True


    

_DISTRIBUTIONS = {
    'Normal': dict(loc=0.0, scale=1.0),
    'Cauchy': dict(loc=0.0, scale=1.0), 
    'Gumbel': dict(loc=1.0, scale=2.0), 
    'Laplace': dict(loc=0.0, scale=1.0), #there's some stuff here for scale \in (0.6, 0.8)
    'Logistic': dict(loc=0.0, scale=1.0),
    # 'Uniform': dict(low=0.0, high=1.0)
}    
dist_names = list(_DISTRIBUTIONS.keys())
DIST_NAME = dist_names[DISTRIBUTION]
VAR_RANGE = torch.linspace(*PARAMS_RANGE, NUM_DISTRIBUTIONS) 
DISTRIBUTIONS = {f'{DIST_NAME}_{var.item():.4f}': dict(loc=0.0, scale=var.item()) for var in VAR_RANGE}

# DIST_NAME = 'Normal'
# VAR_RANGE = torch.linspace(0.90, 1.1, 5)
# DISTRIBUTIONS = {f'{DIST_NAME}_{var.item():.4f}': dict(loc=0, scale=var.item()) for var in VAR_RANGE}

def create_folder_structure(distributions_dict: dict[str, dict[str, float]], root_dir: str = OUTPUT_DIR) -> None:
    for i, distribution_name in enumerate(distributions_dict.keys()):
        
        distribution_outputs = os.path.join(root_dir, distribution_name)
        try:
            os.makedirs(distribution_outputs, exist_ok=True)
        except Exception as e:
            print(e)

# Function to generate a prompt
def generate_prompt(prompt_prefix, artist):
    # Generate the prompt
    prompt = f"{prompt_prefix} {artist}"
    return prompt

def init_txt2img(
        checkpoint_path: str=CHECKPOINT_PATH,
        sampler_name: str='ddim',
        n_steps: int=20,
        ddim_eta: float=0.0,
        autoencoder = None,
        unet_model = None,
        clip_text_embedder = None                                   
        ):
    
    txt2img = Txt2Img(checkpoint_path=checkpoint_path, sampler_name=sampler_name, n_steps=n_steps, ddim_eta=ddim_eta)
    # compute loading time
     

    if FROM_DISK:
        with section("to initialize latent diffusion and load submodels tree"):
            latent_diffusion_model = initialize_latent_diffusion()
            latent_diffusion_model.load_submodel_tree()
            txt2img.initialize_from_model(latent_diffusion_model)
        return txt2img
    else:
        with section("to run `StableDiffusionBaseScript`'s initialization function"):
            txt2img.initialize_script(path=CHECKPOINT_PATH, autoencoder= autoencoder, unet_model = unet_model, clip_text_embedder=clip_text_embedder, force_submodels_init=True)
        
        return txt2img

def get_all_prompts(prompt_prefix, artist_file, num_artists = None):

    with open(artist_file, 'r') as f:
        artists = f.readlines()

    num_seeds = len(NOISE_SEEDS)
    if num_artists is not None:
        artists = artists[:num_artists]
    
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

def generate_images_from_temp_range(
        distributions: dict[str, dict[str, float]], 
        txt2img: Txt2Img,                                      
        output_dir: str = OUTPUT_DIR, 
        clear_output_dir: bool = CLEAR_OUTPUT_DIR,
        prompt_prefix: str="A woman with flowers in her hair in a courtyard, in the style of",
        artist_file: str=os.path.abspath('./input/artists.txt'),
        num_artists: int=1,
        batch_size: int=1,
        temperature_range = TEMP_RANGE,
        ):
    # Clear the output directory

    if clear_output_dir:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    # Create the folder structure for the outputs
    create_folder_structure(distributions, output_dir)
    total_images, prompts = get_all_prompts(prompt_prefix, artist_file, num_artists = 1)
    num_distributions = len(distributions)
    prompts = list(prompts)
    print("num_distributions:", num_distributions)
    noise_seed = NOISE_SEEDS[0]
    prompt = prompts[0]

    # Generate the images
    img_grids = []
    for distribution_index, (distribution_name, params) in enumerate(distributions.items()):
       noise_fn = lambda shape, device = None: get_torch_distribution_from_name(DIST_NAME)(**params).sample(shape).to(device)
       img_rows = []
       for temperature in temperature_range:
        images = txt2img.generate_images(
        batch_size=batch_size,
        prompt=prompt,
        seed=noise_seed,
        noise_fn = noise_fn,
        temperature=temperature.item(),
                    )
        print(temperature.item())
        
        image_name = f"n{noise_seed:04d}_t{temperature}.jpg"
        dest_path = os.path.join(os.path.join(output_dir, distribution_name), image_name)

        print(f"temperature: {temperature}")
        # print("img shape: ", images.shape)
        img_rows.append(images)
        # print("len prompt batch: ", len(prompt_batch))
        save_images(images, dest_path=dest_path)
       row = torch.cat(img_rows, dim=0)
       img_grids.append(row) 
    
    dest_path = os.path.join(output_dir, f"grid_all.jpg")
    
    grid = torch.cat(img_grids, dim=0)
    print("grid shape: ", grid.shape)
    save_image_grid(grid, dest_path, nrow=len(TEMP_RANGE), normalize=True, scale_each=True)

      

def main():
    txt2img = init_txt2img(ddim_eta=DDIM_ETA)
    generate_images_from_temp_range(DISTRIBUTIONS, txt2img, num_artists=NUM_ARTISTS, batch_size=1, temperature_range=TEMP_RANGE)

if __name__ == "__main__":
    main()

# %%
