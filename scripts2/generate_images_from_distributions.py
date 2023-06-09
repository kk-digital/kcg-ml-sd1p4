import os
import sys
import torch
import shutil
import time
import argparse

from typing import Callable
from tqdm import tqdm
base_directory = "./"
sys.path.insert(0, base_directory)

from auxiliary_functions import get_torch_distribution_from_name

from text_to_image import Txt2Img

from stable_diffusion2.latent_diffusion import LatentDiffusion
from stable_diffusion2.stable_diffusion import StableDiffusion
from stable_diffusion2.constants import CHECKPOINT_PATH, AUTOENCODER_PATH, UNET_PATH, EMBEDDER_PATH, LATENT_DIFFUSION_PATH, ENCODER_PATH, DECODER_PATH, TOKENIZER_PATH, TRANSFORMER_PATH
from stable_diffusion2.utils.utils import SectionManager as section
from stable_diffusion2.utils.utils import save_image_grid, save_images
from stable_diffusion2.utils.model import initialize_latent_diffusion, initialize_autoencoder, initialize_clip_embedder
import safetensors.torch as st



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

NUM_ARTISTS = 1
CLEAR_OUTPUT_DIR = True

OUTPUT_DIR = os.path.abspath('./output/noise-tests/')

_DISTRIBUTIONS = {
    'Normal': dict(loc=0.0, scale=1.0),
    'Cauchy': dict(loc=0.0, scale=1.0), 
    'Gumbel': dict(loc=1.0, scale=2.0), 
    'Laplace': dict(loc=0.0, scale=1.0), #there's some stuff here for scale \in (0.6, 0.8)
    'Logistic': dict(loc=0.0, scale=1.0),
    # 'Uniform': dict(low=0.0, high=1.0)
}

parser = argparse.ArgumentParser(
        description='')
parser.add_argument('-fd', '--from_disk', type=bool, default=True)
parser.add_argument('-d', '--distribution', type=int, default=4)
parser.add_argument('--num_distributions', type=int, default=3)
parser.add_argument('--params_range', nargs = "+", type=float, default=[0.49, 0.54])
parser.add_argument('--num_seeds', type=int, default=3)
parser.add_argument('-t', '--temperature', type=float, default=1.0)
parser.add_argument('--ddim_eta', type=float, default=0.0)
parser.add_argument('--latent_diffusion_init_mode', type=int, default=0)
parser.add_argument('--txt2img_init_from_saved', type=bool, default=False)
args = parser.parse_args() 

FROM_DISK = args.from_disk
NUM_SEEDS = args.num_seeds
NOISE_SEEDS = NOISE_SEEDS[:NUM_SEEDS]
DISTRIBUTION = args.distribution
NUM_DISTRIBUTIONS = args.num_distributions
PARAMS_RANGE = args.params_range
TEMPERATURE = args.temperature
DDIM_ETA = args.ddim_eta

dist_names = list(_DISTRIBUTIONS.keys())
DIST_NAME = dist_names[DISTRIBUTION]
VAR_RANGE = torch.linspace(*PARAMS_RANGE, NUM_DISTRIBUTIONS)
DISTRIBUTIONS = {f'{DIST_NAME}_{var.item():.4f}': dict(loc=0.0, scale=var.item()) for var in VAR_RANGE}

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
    
    # txt2img = Txt2Img(checkpoint_path=checkpoint_path, sampler_name=sampler_name, n_steps=n_steps, ddim_eta=ddim_eta)
    stable_diffusion = StableDiffusion(sampler_name=sampler_name, n_steps=n_steps, ddim_eta=ddim_eta)
    

    if FROM_DISK:
        with section("to initialize latent diffusion and load submodels tree"):
            # stable_diffusion.quick_initialize().load_submodel_tree()
            stable_diffusion.load_model()
            stable_diffusion.model.load_submodel_tree()
            stable_diffusion.initialize_sampler()
        return stable_diffusion
    else:
        with section("to run `StableDiffusionBaseScript`'s initialization function"):
            stable_diffusion.initialize_latent_diffusion(path=checkpoint_path, autoencoder= autoencoder, unet_model = unet_model, clip_text_embedder=clip_text_embedder, force_submodels_init=True)
        
        return stable_diffusion

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


def generate_images_from_dist( 
        txt2img: Txt2Img,         
        distribution: tuple[str, dict[str, float]],                             
        output_dir: str = OUTPUT_DIR, 
        prompt_prefix: str="A woman with flowers in her hair in a courtyard, in the style of",
        artist_file: str=os.path.abspath('./input/artists.txt'),
        num_artists: int=2,
        batch_size: int=1,
        temperature: float=1.0,
        ):

    total_images, prompts = get_all_prompts(prompt_prefix, artist_file, num_artists = num_artists)
    dist_name, params = distribution

    with torch.no_grad():
        with tqdm(total=total_images, desc='Generating images', ) as pbar:
            print(f"Generating images for {dist_name}")
            noise_fn = lambda shape, device = None: get_torch_distribution_from_name(DIST_NAME)(**params).sample(shape).to(device)
            grid_rows = []
            img_counter = 0
            for prompt_index, prompt in enumerate(prompts):
                prompt_batch = []
                for seed_index, noise_seed in enumerate(NOISE_SEEDS):
                    p_bar_description = f"Generating image {img_counter+1} of {total_images}. Distribution: {DIST_NAME}{params}"
                    pbar.set_description(p_bar_description)

                    image_name = f"n{noise_seed:04d}_a{prompt_index+1:04d}.jpg"
                    dest_path = os.path.join(os.path.join(output_dir, dist_name), image_name)
                    images = txt2img.generate_images(
                        batch_size=batch_size,
                        prompt=prompt,
                        seed=noise_seed,
                        noise_fn = noise_fn,
                        temperature=temperature,
                    )

                    prompt_batch.append(images)

                    save_images(images, dest_path=dest_path)
                    img_counter += 1

                image_name = f"row_a{prompt_index+1:04d}.jpg"
                dest_path = os.path.join(os.path.join(output_dir, dist_name), image_name)
                row = torch.cat(prompt_batch, dim=0)
                save_image_grid(row, dest_path, normalize=True, scale_each=True)
                grid_rows.append(row)

            dest_path = os.path.join(os.path.join(output_dir, dist_name), f"grid_{dist_name}.jpg")
            grid = torch.cat(grid_rows, dim=0)
            save_image_grid(grid, dest_path, nrow=num_artists, normalize=True, scale_each=True)
            return grid        

def generate_images_from_dist_dict(
        distributions: dict[str, dict[str, float]], 
        txt2img: Txt2Img,                                      
        output_dir: str = OUTPUT_DIR, 
        clear_output_dir: bool = CLEAR_OUTPUT_DIR,
        prompt_prefix: str="A woman with flowers in her hair in a courtyard, in the style of",
        artist_file: str=os.path.abspath('./input/artists.txt'),
        num_artists: int=2,
        batch_size: int=1,
        temperature: float=1.0,
        ):
    
    # Clear the output directory
    if clear_output_dir:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    # Create the folder structure for the outputs
    create_folder_structure(distributions, output_dir)

    num_distributions = len(distributions)
    print("num_distributions:", num_distributions)
    # Generate the images
    img_grids = []
    for distribution_index, (distribution_name, params) in enumerate(distributions.items()):
       grid = generate_images_from_dist(txt2img, (distribution_name, params), prompt_prefix=prompt_prefix, artist_file=artist_file, num_artists=num_artists, batch_size=batch_size, temperature=temperature)
       img_grids.append(grid)
       
    if FROM_DISK:

        dest_path = os.path.join(output_dir, f"grid_all_{DIST_NAME}{VAR_RANGE[0].item():.2f}_{VAR_RANGE[-1].item():.2f}_from_disk.jpg")
    else:
        dest_path = os.path.join(output_dir, f"grid_all_{DIST_NAME}{VAR_RANGE[0].item():.2f}_{VAR_RANGE[-1].item():.2f}.jpg")
    
    grid = torch.cat(img_grids, dim=0)
    torch.save(grid, dest_path.replace('.jpg', '.pt'))
    # st.save_file({"img_grid": grid}, dest_path.replace('.jpg', '.safetensors'))
    save_image_grid(grid, dest_path, nrow=NUM_SEEDS, normalize=True, scale_each=True)



def main():
    txt2img = init_txt2img(ddim_eta=DDIM_ETA)
    generate_images_from_dist_dict(DISTRIBUTIONS, txt2img, num_artists=NUM_ARTISTS, batch_size=1, temperature=TEMPERATURE)

if __name__ == "__main__":
    main()
