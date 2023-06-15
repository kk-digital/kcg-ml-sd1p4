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

# noise_seeds = [
#     2982,
#     4801,
#     1995,
#     3598,
#     987,
#     3688,
#     8872,
#     762
# ]
noise_seeds = [
    2982,
    762
]
#

if len(sys.argv) == 1:
    DISTRIBUTIONS = {
        'Normal': dict(loc=0.0, scale=1.0),
        'Cauchy': dict(loc=0.0, scale=1.0), 
        'Gumbel': dict(loc=1.0, scale=2.0), 
        'Laplace': dict(loc=0.0, scale=1.0), 
        'Uniform': dict(low=0.0, high=1.0)
    }
else:
    VAR_RANGE = torch.linspace(0.95, 1.05, 10)
    DISTRIBUTIONS = {f'Normal_{var.item():.4f}': dict(loc=0, scale=var.item()) for var in VAR_RANGE}

TEMPERATURE = 1.5
DDIM_ETA = -0.25
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
    for i, distribution_name in enumerate(distributions_dict.keys()):
        
        if len(sys.argv) > 1:
            # for var in VAR_RANGE:
            #     distribution_outputs = os.path.join(root_dir, distribution_name+f'-mu0-sigma{var.item():.2f}')
            #     try:
            #         os.makedirs(distribution_outputs, exist_ok=True)
            #     except Exception as e:
            #         print(e)
            # distribution_outputs = os.path.join(root_dir, distribution_name+f'-mu0-sigma{VAR_RANGE[i].item():.2f}')
            distribution_outputs = os.path.join(root_dir, distribution_name)
        else:
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

def init_txt2img(checkpoint_path, sampler_name, n_steps, ddim_eta=0.0):
    txt2img = Txt2Img(checkpoint_path=checkpoint_path, sampler_name=sampler_name, n_steps=n_steps, ddim_eta=ddim_eta)
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


import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import make_grid
from types import FunctionType
from typing import Any, BinaryIO, List, Optional, Tuple, Union
import pathlib
from PIL import Image, ImageColor, ImageDraw, ImageFont
import numpy as np

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

def norm_torch(x_all):
    # runs unity norm on all timesteps of all samples
    # input is (n_samples, 3,h,w), the torch image format
    x = x_all.cpu().numpy()
    xmax = x.max((2,3))
    xmin = x.min((2,3))
    xmax = np.expand_dims(xmax,(2,3)) 
    xmin = np.expand_dims(xmin,(2,3))
    nstore = (x - xmin)/(xmax - xmin)
    return torch.from_numpy(nstore)

def plot_grid(x,n_sample,n_rows,save_dir,w):
    # x:(n_sample, 3, h, w)
    ncols = n_sample//n_rows
    grid = make_grid(norm_torch(x), nrow=ncols)  # curiously, nrow is number of columns.. or number of items in the row.
    save_image(grid, save_dir + f"run_image_w{w}.png")
    print('saved image at ' + save_dir + f"run_image_w{w}.png")
    return grid

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
        ddim_eta: float=0.0,
        temperature: float=1.0,                                      
        ):


    # Clear the output directory
    if clear_output_dir:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    # Create the folder structure for the outputs
    create_folder_structure(distributions, output_dir)

    time_before_initialization = time.time()
    
    txt2img = init_txt2img(checkpoint_path, sampler_name, n_steps, ddim_eta=ddim_eta)

    time_after_initialization = time.time()

    total_images, prompts = get_all_prompts(prompt_prefix, artist_file)
    
    num_prompts_per_distribution = 2

    # Generate the images
    if len(sys.argv) > 1:
        with torch.no_grad():
            with tqdm(total=total_images, desc='Generating images', ) as pbar:
                print("distributions itens: ", len(distributions.items()), list(distributions.items()))
                for distribution_name, params in distributions.items():
                    print(distribution_name, params)
                    counter = 0
                    print(f"Generating images for {distribution_name}")
                    noise_fn = lambda shape, device = None: get_torch_distribution_from_name('Normal')(**params).sample(shape).to(device)
                    grid_columns = []
                    for prompt_index, prompt in enumerate(prompts):
                        prompt_batch = []
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
                                    temperature=temperature,
                                )
                                # print("img shape: ", images.shape)
                                prompt_batch.append(images)
                                # print("len prompt batch: ", len(prompt_batch))
                                save_images(images, dest_path=dest_path)
                                
                                pbar.update(1)
                                # TODO adjust this loop so that it generates a fixed number of images per distribution instead of all images, less uglily
                            counter += 1
                            image_name = f"grid_a{prompt_index+1:04d}.jpg"
                            dest_path = os.path.join(os.path.join(output_dir, distribution_name), image_name)
                            print("len prompt batch: ", len(prompt_batch))
                            print([img.shape for img in prompt_batch])
                            column = torch.cat(prompt_batch, dim=0)
                            print("col shape: ", column.shape)
                            save_image(column, dest_path)
                            print("counter: ", counter)
                        else:

                            counter = 0
                            break
                        # grid_columns.append(torch.cat(prompt_batch, dim=0))
                        # print("grid_columns: ", [column.shape for column in grid_columns])
                    # grid = torch.cat(grid_columns, dim=0)
                    # print("grid: ", grid.shape)
                    # grid = make_grid(grid_columns)
                    # dest_path = os.path.join(os.path.join(output_dir, distribution_name), "grid.jpg")
                    # save_images(grid_columns, dest_path)
    else:
        with torch.no_grad():
            with tqdm(total=total_images, desc='Generating images', ) as pbar:
                print("distributions itens: ", len(distributions.items()))
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
    generate_images_from_custom_noise(DISTRIBUTIONS, ddim_eta=DDIM_ETA, temperature=TEMPERATURE)

if __name__ == "__main__":
    main()

# %%
