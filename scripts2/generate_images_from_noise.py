import os
import sys
import torch
import shutil
import time
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
FROM_DISK = ((os.sys.argv[2] == 'True') if len(sys.argv) > 2 else False) 
DISK_MODE = None #1 or 2
if FROM_DISK:
    if len(os.sys.argv) > 3:
        DISK_MODE = int(os.sys.argv[3])

CLEAR_OUTPUT_DIR = True
if len(sys.argv) > 1:
    dist_name_index = int(sys.argv[1])

    _DISTRIBUTIONS = {
        'Normal': dict(loc=0.0, scale=1.0),
        'Cauchy': dict(loc=0.0, scale=1.0), 
        'Gumbel': dict(loc=1.0, scale=2.0), 
        'Laplace': dict(loc=0.0, scale=1.0), #there's some stuff here for scale \in (0.6, 0.8)
        'Logistic': dict(loc=0.0, scale=1.0),
        # 'Uniform': dict(low=0.0, high=1.0)
    }
    
    dist_names = list(_DISTRIBUTIONS.keys())
    DIST_NAME = dist_names[dist_name_index]
    #VAR_RANGE = torch.linspace(0.6, 0.8, 10) #args here should be given as command line arguments
    VAR_RANGE = torch.linspace(0.49, 0.54, NUM_DISTRIBUTIONS) #args here should be given as command line arguments
    DISTRIBUTIONS = {f'{DIST_NAME}_{var.item():.4f}': dict(loc=0.0, scale=var.item()) for var in VAR_RANGE}
else:
    DIST_NAME = 'Normal'
    VAR_RANGE = torch.linspace(0.90, 1.1, 5)
    DISTRIBUTIONS = {f'{DIST_NAME}_{var.item():.4f}': dict(loc=0, scale=var.item()) for var in VAR_RANGE}

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
        if DISK_MODE == 1:
            txt2img.initialize_from_saved(model_path=LATENT_DIFFUSION_PATH)
        elif DISK_MODE == 2:
            with section("to load all submodels from disk"):
                # autoencoder = torch.load(AUTOENCODER_PATH)
                # autoencoder.eval()
                # clip_text_embedder = torch.load(EMBEDDER_PATH)
                # clip_text_embedder.eval()
                # unet_model = torch.load(UNET_PATH)
                # unet_model.eval()

                # latent_diffusion_model.load_submodels()

            # latent_diffusion_model = LatentDiffusion(linear_start=0.00085,
            #                     linear_end=0.0120,
            #                     n_steps=1000,
            #                     latent_scaling_factor=0.18215,
            #                     autoencoder=autoencoder,
            #                     clip_embedder=clip_text_embedder,
            #                     unet_model=unet_model)
            # summary(model)
                latent_diffusion_model = initialize_latent_diffusion(path=CHECKPOINT_PATH, force_submodels_init=True)
                latent_diffusion_model.save_submodels()
                latent_diffusion_model.first_stage_model.save_submodels()
                latent_diffusion_model.cond_stage_model.save_submodels()
            # with section(f"stable-diffusion checkpoint loading, from {CHECKPOINT_PATH}"):
            #     checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

            # # Set model state
            # with section('model state loading'):
                # missing_keys, extra_keys = latent_diffusion_model.load_state_dict(checkpoint["state_dict"], strict=False)
                # latent_diffusion_model.eval()
            txt2img.initialize_from_model(latent_diffusion_model)

        elif DISK_MODE == 3:
            with section("to load latent diffusion saved model"):
                autoencoder = torch.load(AUTOENCODER_PATH)
                autoencoder.eval()
                clip_text_embedder = torch.load(EMBEDDER_PATH)
                clip_text_embedder.eval()
                unet_model = torch.load(UNET_PATH)
                unet_model.eval()
                latent_diffusion_model = initialize_latent_diffusion(path=None, autoencoder=autoencoder, unet_model = unet_model, clip_text_embedder=clip_text_embedder, force_submodels_init=False)
                # latent_diffusion_model.load_submodels()
            txt2img.initialize_from_model(latent_diffusion_model)


        elif DISK_MODE == 4:
            with section("to load latent diffusion saved model"):
                autoencoder = torch.load(AUTOENCODER_PATH)
                autoencoder.eval()
                clip_text_embedder = torch.load(EMBEDDER_PATH)
                clip_text_embedder.eval()
                unet_model = torch.load(UNET_PATH)
                unet_model.eval()
                # print(type(latent_diffusion_model))
                # latent_diffusion_model.load_submodels()
            txt2img.initialize_script(autoencoder=autoencoder, unet_model = unet_model, clip_text_embedder=clip_text_embedder, force_submodels_init=False)

        elif DISK_MODE == 5: #builds the autoencoder and the embedder from submodels before loading the latent diffusion model
            with section("to load latent diffusion saved model"):
                # encoder = torch.load(ENCODER_PATH)
                # encoder.eval()
                # decoder = torch.load(DECODER_PATH)
                # decoder.eval()
                # autoencoder = initialize_autoencoder(encoder=encoder, decoder=decoder, force_submodels_init=False)
                autoencoder = torch.load(AUTOENCODER_PATH)
                autoencoder.eval()                
                tokenizer = torch.load(TOKENIZER_PATH)
                transformer = torch.load(TRANSFORMER_PATH)
                transformer.eval()
                clip_text_embedder = initialize_clip_embedder(tokenizer=tokenizer, transformer=transformer, force_submodels_init=False)
                
                unet_model = torch.load(UNET_PATH)
                unet_model.eval()
                # print(type(latent_diffusion_model))
                # latent_diffusion_model.load_submodels()
            txt2img.initialize_script(autoencoder=autoencoder, unet_model = unet_model, clip_text_embedder=clip_text_embedder, force_submodels_init=False)

        elif DISK_MODE == 6: #builds the autoencoder and the embedder from submodels before loading the latent diffusion model
            with section("to load latent diffusion saved model"):
                encoder = torch.load(ENCODER_PATH)
                encoder.eval()
                decoder = torch.load(DECODER_PATH)
                decoder.eval()
                autoencoder = initialize_autoencoder(encoder=encoder, decoder=decoder, force_submodels_init=False)
                # autoencoder = torch.load(AUTOENCODER_PATH)
                autoencoder.eval()                
                # autoencoder = initialize_autoencoder(encoder=encoder, decoder=decoder, force_submodels_init=False)

                tokenizer = torch.load(TOKENIZER_PATH)
                transformer = torch.load(TRANSFORMER_PATH)
                transformer.eval()
                clip_text_embedder = initialize_clip_embedder(tokenizer=tokenizer, transformer=transformer, force_submodels_init=False)
                
                unet_model = torch.load(UNET_PATH)
                unet_model.eval()
                latent_diffusion_model = initialize_latent_diffusion(path=None, autoencoder=autoencoder, unet_model = unet_model, clip_text_embedder=clip_text_embedder, force_submodels_init=False)
                print(type(latent_diffusion_model))
                # latent_diffusion_model.load_submodels()
            txt2img.initialize_from_model(latent_diffusion_model)

        elif DISK_MODE == 7: #builds the latent diffusion model then loads submodels
            with section("to load latent diffusion saved model"):

                latent_diffusion_model = initialize_latent_diffusion(path=None, autoencoder=autoencoder, unet_model = unet_model, clip_text_embedder=clip_text_embedder, force_submodels_init=False)               
                print(type(latent_diffusion_model))
                latent_diffusion_model.load_submodels()
            txt2img.initialize_from_model(latent_diffusion_model)

        return txt2img
    else:
        with section("to run `StableDiffusionBaseScript`'s initialization function"):
            txt2img.initialize_script(autoencoder= autoencoder, unet_model = unet_model, clip_text_embedder=clip_text_embedder)
        
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
       
    if FROM_DISK:
        if type(DISK_MODE) is int:
            dest_path = os.path.join(output_dir, f"grid_all_{DIST_NAME}{VAR_RANGE[0].item():.2f}_{VAR_RANGE[-1].item():.2f}_from_disk_m{DISK_MODE}.jpg")
    else:
        dest_path = os.path.join(output_dir, f"grid_all_{DIST_NAME}{VAR_RANGE[0].item():.2f}_{VAR_RANGE[-1].item():.2f}.jpg")
    
    grid = torch.cat(img_grids, dim=0)
    torch.save(grid, dest_path.replace('.jpg', '.pt'))
    st.save_file({"img_grid": grid}, dest_path.replace('.jpg', '.safetensors'))
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
