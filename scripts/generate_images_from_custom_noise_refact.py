#%%


import os
import sys
from typing import Callable
from custom_noise.text_to_image_custom import Txt2Img
import torch
import time
import shutil
from tqdm import tqdm
import torchvision
from stable_diffusion.utils.model import save_images
from cli_builder import CLI
from torchvision.utils import make_grid
from typing import BinaryIO, List, Optional, Union
import pathlib
from PIL import Image

noise_seeds = [
    2982,
    4801,
    1995,
    3598,
    # 987,
    # 3688,
    # 8872,
    # 762
]
# noise_seeds = [
#     2982,
# ]
#

NUM_SEEDS = len(noise_seeds)

if len(sys.argv) > 1:
    dist_name_index = int(sys.argv[1])

    DISTRIBUTIONS = {
        'Normal': dict(loc=0.0, scale=1.0),
        'Cauchy': dict(loc=0.0, scale=1.0), 
        'Gumbel': dict(loc=1.0, scale=2.0), 
        'Laplace': dict(loc=0.0, scale=1.0), #there's some stuff here for scale \in (0.6, 0.8)
        'Logistic': dict(loc=0.0, scale=1.0),
        # 'Uniform': dict(low=0.0, high=1.0)
    }
    
    dist_names = list(DISTRIBUTIONS.keys())
    DIST_NAME = dist_names[dist_name_index]
    #VAR_RANGE = torch.linspace(0.6, 0.8, 10) #args here should be given as command line arguments
    VAR_RANGE = torch.linspace(0.49, 0.54, 20) #args here should be given as command line arguments
    DISTRIBUTIONS = {f'{DIST_NAME}_{var.item():.4f}': dict(loc=0.0, scale=var.item()) for var in VAR_RANGE}
else:
    DIST_NAME = 'Normal'
    VAR_RANGE = torch.linspace(0.90, 1.1, 5)
    DISTRIBUTIONS = {f'{DIST_NAME}_{var.item():.4f}': dict(loc=0, scale=var.item()) for var in VAR_RANGE}

TEMPERATURE = 1.0 #should be cli argument
DDIM_ETA = 0.0 #should be cli argument
OUTPUT_DIR = os.path.abspath('./output/noise-tests/')

CLEAR_OUTPUT_DIR = True
NUM_ARTISTS = 1
def get_all_torch_distributions() -> tuple[list[str], list[type]]:
    
    torch_distributions_names = torch.distributions.__all__
    
    torch_distributions = [
        torch.distributions.__dict__[torch_distribution_name] \
            for torch_distribution_name in torch_distributions_names
            ]
    
    return torch_distributions_names, torch_distributions

def get_torch_distribution_from_name(name: str) -> type:
    if name == 'Logistic':
        def logistic_distribution(loc, scale):
            base_distribution = torch.distributions.Uniform(0, 1)
            transforms = [torch.distributions.transforms.SigmoidTransform().inv, torch.distributions.transforms.AffineTransform(loc=loc, scale=scale)]
            logistic = torch.distributions.TransformedDistribution(base_distribution, transforms)
            return logistic
        return logistic_distribution
    return torch.distributions.__dict__[name]

def build_noise_samplers(distributions: dict[str, dict[str, float]]) -> dict[str, Callable]:
    noise_samplers = {
        k: lambda shape, device = None: get_torch_distribution_from_name(k)(**v).sample(shape).to(device) \
                       for k, v in distributions.items()
                       }
    return noise_samplers

def create_folder_structure(distributions_dict: dict[str, dict[str, float]], root_dir: str = OUTPUT_DIR) -> None:
    for i, distribution_name in enumerate(distributions_dict.keys()):
        
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

def init_txt2img(
        checkpoint_path: str=os.path.abspath('./input/model/sd-v1-4.ckpt'),
        sampler_name: str='ddim',
        n_steps: int=20,
        ddim_eta: float=0.0,                                   
        ):
    txt2img = Txt2Img(checkpoint_path=checkpoint_path, sampler_name=sampler_name, n_steps=n_steps, ddim_eta=ddim_eta)
    txt2img.initialize_script()
    return txt2img

def get_all_prompts(prompt_prefix, artist_file, num_artists = None):

    with open(artist_file, 'r') as f:
        artists = f.readlines()

    num_seeds = len(noise_seeds)
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



def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)


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
                for seed_index, noise_seed in enumerate(noise_seeds):
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
                save_image(row, dest_path, normalize=True, scale_each=True)
                grid_rows.append(row)
            dest_path = os.path.join(os.path.join(output_dir, dist_name), f"grid_{dist_name}.jpg")
            print("grid_rows: ", [row.shape for row in grid_rows])
            grid = torch.cat(grid_rows, dim=0)
            print("grid shape: ", grid.shape)
            save_image(grid, dest_path, nrow=num_artists, normalize=True, scale_each=True)
            return grid    
            # save_image(grid_rows, dest_path, nrow=num_artists, normalize=True, scale_each=True)    

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
    
    dest_path = os.path.join(output_dir, f"grid_all.jpg")
    
    grid = torch.cat(img_grids, dim=0)
    print("grid shape: ", grid.shape)
    save_image(grid, dest_path, nrow=NUM_SEEDS, normalize=True, scale_each=True)
  
      

def main():
    txt2img = init_txt2img(ddim_eta=DDIM_ETA)
    generate_images_from_dist_dict(DISTRIBUTIONS, txt2img, num_artists=NUM_ARTISTS, batch_size=1, temperature=TEMPERATURE)

if __name__ == "__main__":
    main()

# %%
