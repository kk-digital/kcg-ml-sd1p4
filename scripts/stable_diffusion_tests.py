import os
import sys
import torch
import shutil
import time
from typing import Callable
from tqdm import tqdm
import argparse

from auxiliary_functions import get_torch_distribution_from_name
from text_to_image import Txt2Img

from stable_diffusion.StableDiffusionBase import LatentDiffusion
from stable_diffusion.UnetDiffusionModelWrapper import UnetDiffusionModelWrapper
# from stable_diffusion.latent_diffusion_model import LatentDiffusionModel
from stable_diffusion.constants import CHECKPOINT_PATH, AUTOENCODER_PATH, UNET_PATH, EMBEDDER_PATH, LATENT_DIFFUSION_PATH
from labml.monit import section
from stable_diffusion.utils.utils import save_images, save_image_grid
from stable_diffusion.utils.model import initialize_autoencoder, initialize_encoder, initialize_decoder, check_device, get_device
from stable_diffusion.utils.model import initialize_clip_embedder, initialize_tokenizer, initialize_transformer 
from stable_diffusion.utils.model import initialize_unet, initialize_latent_diffusion
import safetensors.torch as st

# CHECKPOINT_PATH = os.path.abspath('./input/model/v1-5-pruned-emaonly.ckpt')

CHECKPOINT_PATH = os.path.abspath('./input/model/v1-5-pruned-emaonly.ckpt')

EMBEDDER_PATH = os.path.abspath('./input/model/clip/clip_embedder.ckpt')
TOKENIZER_PATH = os.path.abspath('./input/model/clip/clip_tokenizer.ckpt')
TRANSFORMER_PATH = os.path.abspath('./input/model/clip/clip_transformer.ckpt')

UNET_PATH = os.path.abspath('./input/model/unet/unet.ckpt')

AUTOENCODER_PATH = os.path.abspath('./input/model/autoencoder/autoencoder.ckpt')
ENCODER_PATH = os.path.abspath('./input/model/autoencoder/encoder.ckpt')
DECODER_PATH = os.path.abspath('./input/model/autoencoder/decoder.ckpt')

LATENT_DIFFUSION_PATH = os.path.abspath('./input/model/latent_diffusion/latent_diffusion.ckpt')


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
# NOISE_SEEDS = [
#     2982,
# ]
#

NUM_SEEDS = 3
NOISE_SEEDS = NOISE_SEEDS[:NUM_SEEDS]
NUM_DISTRIBUTIONS = 3
NUM_ARTISTS = 1
TEMPERATURE = 1.0 #should be cli argument
TEMP_RANGE = torch.linspace(1, 4, 8)
DDIM_ETA = 0.0 #should be cli argument
OUTPUT_DIR = os.path.abspath('./output/noise-tests/')
TESTS_OUTPUT_DIR = os.path.abspath('./output/outputs_for_test/')
parser = argparse.ArgumentParser(
        description='')
parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
parser.add_argument('--vae_init_mode', type=int, default=0)
parser.add_argument('--clip_init_mode', type=int, default=0)
parser.add_argument('--latent_diffusion_init_mode', type=int, default=0)
parser.add_argument('--default', type=bool, default=False)
parser.add_argument('--alternative', type=bool, default=False)
args = parser.parse_args()
print(args)
VAE_INIT_MODE = args.vae_init_mode
CLIP_INIT_MODE = args.clip_init_mode
LATENT_DIFFUSION_INIT_MODE = args.latent_diffusion_init_mode
DEFAULT = args.default
ALT = args.alternative


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
DIST_NAME = 'Logistic'
#VAR_RANGE = torch.linspace(0.6, 0.8, 10) #args here should be given as command line arguments
VAR_RANGE = torch.linspace(0.49, 0.54, NUM_DISTRIBUTIONS) #args here should be given as command line arguments
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


def init_vae_from_mode(mode):
    if mode == 0:
        with section("to initialize autoencoder, then load encoder and decoder from disk"):
            autoencoder = initialize_autoencoder(force_submodels_init=False)
            autoencoder.load_submodels()
            return autoencoder
            
    elif mode == 1:
        with section("to initialize encoder and decoder, then initialize autoencoder from them"):
            encoder = initialize_encoder()
            decoder = initialize_decoder()
            autoencoder = initialize_autoencoder(encoder=encoder, decoder=decoder)    
            return autoencoder
        
def init_text_embedder_from_mode(mode):
    if mode == 0:
        with section("to initialize text embedder, then load tokenizer and transformer from disk"):
            clip_embedder = initialize_clip_embedder(force_submodels_init=False)
            clip_embedder.load_submodels()
            return clip_embedder
            
    elif mode == 1:
        with section("to initialize encoder and decoder, then initialize autoencoder from them"):
            tokenizer = initialize_tokenizer()
            transformer = initialize_transformer()
            clip_embedder = initialize_clip_embedder(tokenizer = tokenizer, transformer=transformer)
            return clip_embedder

def init_latent_diffusion_from_mode(mode):
    if mode == 0:
        with section("to initialize latent diffusion, then load submodels from disk"):
            latent_diffusion_model = initialize_latent_diffusion(path = CHECKPOINT_PATH, force_submodels_init=False)
            latent_diffusion_model.load_submodels()
            return latent_diffusion_model
    if mode == 1:
        with section("to initialize latent diffusion submodels, then initialize latent diffusion from them"):
            autoencoder = init_vae_from_mode(VAE_INIT_MODE)
            clip_text_embedder = init_text_embedder_from_mode(CLIP_INIT_MODE)
            unet_model = initialize_unet()
            latent_diffusion_model = initialize_latent_diffusion(path = CHECKPOINT_PATH, autoencoder=autoencoder, clip_text_embedder=clip_text_embedder, unet_model=unet_model)
            return latent_diffusion_model
    if mode == 2:
        with section("to initialize latent diffusion forcing submodels initialization"):
            latent_diffusion_model = initialize_latent_diffusion(path = CHECKPOINT_PATH, force_submodels_init=True)
            return latent_diffusion_model
    if mode == 3:
        with section("to load latent diffusion from disk, then load its submodels from disk"):
            latent_diffusion_model = torch.load(LATENT_DIFFUSION_PATH)
            latent_diffusion_model.load_submodels()
            latent_diffusion_model.autoencoder.load_submodels()
            latent_diffusion_model.clip_embedder.load_submodels()
            latent_diffusion_model.eval()
            return latent_diffusion_model        
    if mode == 4:
        with section("to initialize latent diffusion empty and without loading weights, with external initialization function, then load its submodels from disk"):
            latent_diffusion_model = initialize_latent_diffusion(force_submodels_init=False)
            latent_diffusion_model.load_submodels()
            latent_diffusion_model.autoencoder.load_submodels()
            latent_diffusion_model.clip_embedder.load_submodels()
            latent_diffusion_model.eval()
            return latent_diffusion_model 
    if mode == 5:
        with section("to initialize latent diffusion empty and without loading weights, then load its submodels from disk"):
            device = check_device(0)
            latent_diffusion_model = LatentDiffusion(linear_start=0.00085,
                                linear_end=0.0120,
                                n_steps=1000,
                                latent_scaling_factor=0.18215
                                ).to(device)
            latent_diffusion_model.load_submodels()
            latent_diffusion_model.autoencoder.load_submodels()
            latent_diffusion_model.clip_embedder.load_submodels()
            latent_diffusion_model.eval()
            return latent_diffusion_model 
    if mode == 6:
        with section("to load latent diffusion saved model"):
            autoencoder = torch.load(AUTOENCODER_PATH)
            autoencoder.eval()
            autoencoder.load_submodels()
            clip_text_embedder = torch.load(EMBEDDER_PATH)
            clip_text_embedder.eval()
            clip_text_embedder.load_submodels()
            unet_model = torch.load(UNET_PATH)
            unet_model.eval()
            latent_diffusion_model = initialize_latent_diffusion(path=None, autoencoder=autoencoder, unet_model = unet_model, clip_text_embedder=clip_text_embedder, force_submodels_init=False)
            # latent_diffusion_model.load_submodels()
            return latent_diffusion_model   
    if mode == 7:
        with section("to load each submodel, initialize latent diffusion without them, then assigning them to latent diffusion fields"):
            autoencoder = torch.load(AUTOENCODER_PATH)
            autoencoder.eval()
            autoencoder.load_submodels()
            clip_text_embedder = torch.load(EMBEDDER_PATH)
            clip_text_embedder.eval()
            clip_text_embedder.load_submodels()
            unet_model = torch.load(UNET_PATH)
            unet_model.eval()
            device = check_device(0)
            latent_diffusion_model = LatentDiffusion(linear_start=0.00085,
                                linear_end=0.0120,
                                n_steps=1000,
                                latent_scaling_factor=0.18215
                                ).to(device)
            latent_diffusion_model.autoencoder = autoencoder
            latent_diffusion_model.clip_embedder = clip_text_embedder
            latent_diffusion_model.model = DiffusionWrapper(unet_model)
            # latent_diffusion_model.load_submodels()
            return latent_diffusion_model
    if mode == 8:
        with section("to initialize latent diffusion empty and without weights, then loading the submodels tree from disk"):
            device = check_device(0)
            latent_diffusion_model = LatentDiffusion(linear_start=0.00085,
                                linear_end=0.0120,
                                n_steps=1000,
                                latent_scaling_factor=0.18215
                                ).to(device)
            latent_diffusion_model.load_submodel_tree()
            return latent_diffusion_model        



def init_txt2img(
        checkpoint_path: str=CHECKPOINT_PATH,
        sampler_name: str='ddim',
        n_steps: int=20,
        ddim_eta: float=0.0,                                
        ):
    
    txt2img = Txt2Img(checkpoint_path=checkpoint_path, sampler_name=sampler_name, n_steps=n_steps, ddim_eta=ddim_eta)
    # compute loading time

    if DEFAULT:
        with section("to default init"):
            device = check_device(None)
            latent_diffusion_model = LatentDiffusion(linear_start=0.00085,
                linear_end=0.0120,
                n_steps=1000,
                latent_scaling_factor=0.18215
                ).to(device)
            latent_diffusion_model.load_submodel_tree(device = device)
            txt2img.initialize_from_model(latent_diffusion_model)
        return txt2img
    elif ALT:
        with section("to alt init"):
            device = check_device(None)
            latent_diffusion_model = LatentDiffusionModel(linear_start=0.00085,
                linear_end=0.0120,
                n_steps=1000,
                latent_scaling_factor=0.18215
                )
            latent_diffusion_model.alpha_bar.to(device)
            latent_diffusion_model.beta.to(device)
            latent_diffusion_model.load_submodel_tree(device = device)
            txt2img.initialize_from_model(latent_diffusion_model)
        return txt2img
    else:
        latent_diffusion_model = init_latent_diffusion_from_mode(LATENT_DIFFUSION_INIT_MODE)
        txt2img.initialize_from_model(latent_diffusion_model)
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

    # noise_fn = lambda shape, device = None: get_torch_distribution_from_name(dist_name)(**params).sample(shape).to(device)
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
                    print(f"temperature: {temperature}")
                    # print("img shape: ", images.shape)
                    prompt_batch.append(images)
                    # print("len prompt batch: ", len(prompt_batch))
                    save_images(images, dest_path=dest_path)
                    img_counter += 1

                image_name = f"row_a{prompt_index+1:04d}.jpg"
                dest_path = os.path.join(os.path.join(output_dir, dist_name), image_name)
                print("len prompt batch: ", len(prompt_batch))
                print([img.shape for img in prompt_batch])
                row = torch.cat(prompt_batch, dim=0)
                print("row shape: ", row.shape)
                save_image_grid(row, dest_path, normalize=True, scale_each=True)
                grid_rows.append(row)
            dest_path = os.path.join(os.path.join(output_dir, dist_name), f"grid_{dist_name}.jpg")
            print("grid_rows: ", [row.shape for row in grid_rows])
            grid = torch.cat(grid_rows, dim=0)
            print("grid shape: ", grid.shape)
            save_image_grid(grid, dest_path, nrow=num_artists, normalize=True, scale_each=True)
            return grid    
            # save_image_grid(grid_rows, dest_path, nrow=num_artists, normalize=True, scale_each=True)    

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
       

    dest_path = os.path.join(TESTS_OUTPUT_DIR, f"grid_all_{DIST_NAME}{VAR_RANGE[0].item():.2f}_{VAR_RANGE[-1].item():.2f}_{VAE_INIT_MODE}{CLIP_INIT_MODE}{LATENT_DIFFUSION_INIT_MODE}.jpg")
    
    grid = torch.cat(img_grids, dim=0)
    torch.save(grid, dest_path.replace('.jpg', '.pt'))
    # st.save_file({"img_grid": grid}, dest_path.replace('.jpg', '.safetensors'))
    print("grid shape: ", grid.shape)
    save_image_grid(grid, dest_path, nrow=NUM_SEEDS, normalize=True, scale_each=True)

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
        print(temperature)
        
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
    # generate_images_from_temp_range(DISTRIBUTIONS, txt2img, num_artists=NUM_ARTISTS, batch_size=1, temperature_range=TEMP_RANGE)
    generate_images_from_dist_dict(DISTRIBUTIONS, txt2img, num_artists=NUM_ARTISTS, batch_size=1, temperature=TEMPERATURE)

if __name__ == "__main__":
    main()

# %%
