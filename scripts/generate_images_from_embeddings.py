import argparse
import os
import shutil
import sys
from os.path import join
from random import randrange
import torch
from tqdm import tqdm

base_directory = "./"
sys.path.insert(0, base_directory)

from stable_diffusion.stable_diffusion import StableDiffusion
from stable_diffusion.utils_backend import get_device
from stable_diffusion.utils_image import save_images, save_image_grid
from utility.utils_argument_parsing import get_seed_array_from_string

# CHECKPOINT_PATH = os.path.abspath('./input/model/v1-5-pruned-emaonly.ckpt')

OUTPUT_DIR = os.path.abspath("./output/noise-tests/from_embeddings")
EMBEDDED_PROMPTS_DIR = os.path.abspath("./input/embedded_prompts/")

def parse_arguments():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-p",
        "--embedded_prompts_dir",
        type=str,
        default=EMBEDDED_PROMPTS_DIR,
        help="The path to the directory containing the embedded prompts tensors. Defaults to EMBEDDED_PROMPTS_DIR constant, which should be './input/embedded_prompts/'",
    )
    parser.add_argument(
        "-od",
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="The output directory. defaults to OUTPUT_DIR constant, which should be './output/noise-tests/from_embeddings'",
    )
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("-bs", "--batch_size", type=int, default=1)
    parser.add_argument("-t", "--temperature", type=float, default=1.0)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--clear_output_dir", type=bool, default=False)
    parser.add_argument("--cuda_device", type=str, default=None)
    parser.add_argument('--low_vram', action='store_true',
                        help='Low vram means the batch size will be allways 1')
    parser.add_argument("--sampler", type=str, default="ddim")
    parser.add_argument("--cfg_scale", type=float, default=7.0)
    parser.add_argument("--seed", type=str, default="")

    args = parser.parse_args()
    return args;



def init_stable_diffusion(device, sampler_name="ddim", n_steps=20, ddim_eta=0.0):
    device = get_device(device)

    stable_diffusion = StableDiffusion(
        device=device, sampler_name=sampler_name, n_steps=n_steps, ddim_eta=ddim_eta
    )

    stable_diffusion.quick_initialize()
    stable_diffusion.model.load_unet()
    stable_diffusion.model.load_autoencoder().load_decoder()

    return stable_diffusion


def generate_images_from_embeddings(
        embeddings_dir: str = EMBEDDED_PROMPTS_DIR,
        output_dir: str = OUTPUT_DIR,
        sampler_name: str = "ddim",
        n_steps: int = 20,
        batch_size: int = 1,
        num_images : int = 1,
        seed_array: list = 0,
        clear_output_dir: bool = False,
        ddim_eta: float = 0,
        cfg_scale: float = 7.5,
        temperature : float = 1.0,
        cuda_device: str = "cuda",
):
    null_cond = torch.load(join(embeddings_dir, "null_cond.pt"), map_location=cuda_device)
    # print(null_cond.shape)
    embeddings = torch.load(
        join(embeddings_dir, "embedded_prompts.pt"), map_location=cuda_device
    )

    # for embedding in embeddings:
    #     print(embedding.shape)
    stable_diffusion = init_stable_diffusion(
        device=cuda_device,
        sampler_name=sampler_name,
        n_steps=n_steps,
        ddim_eta=ddim_eta,
    )

    if clear_output_dir:
        try:
            shutil.rmtree(output_dir)
        except Exception as e:
            print(e)

    os.makedirs(output_dir, exist_ok=True)

    total_images = len(seed_array) * len(embeddings)

    print("seed array ", seed_array);
    with torch.no_grad():
        with tqdm(
                total=total_images,
                desc="Generating images",
        ) as pbar:
            img_count = 0
            img_grid = []
            for prompt_index, prompt in enumerate(embeddings):
                img_row = []
                for seed_index, noise_seed in enumerate(seed_array):
                    print("noise_seed ", type(noise_seed), str(noise_seed))
                    p_bar_description = (
                        f"Generating image {img_count + 1} of {total_images}"
                    )
                    pbar.set_description(p_bar_description)

                    image_name = f"{prompt_index }.jpg"
                    dest_path = join(output_dir, image_name)

                    latent = stable_diffusion.generate_images_latent_from_embeddings(
                        batch_size=batch_size,
                        embedded_prompt=prompt.unsqueeze(0),
                        null_prompt=null_cond,
                        seed=noise_seed,
                        temperature=temperature,
                        uncond_scale=cfg_scale
                    )
                    images = stable_diffusion.get_image_from_latent(latent)
                    # print(images.shape)
                    save_images(images, dest_path=dest_path)
                    img_row.append(images)
                    img_count += 1
                    pbar.update(1)
                img_grid.append(torch.cat(img_row, dim=0))
            save_image_grid(
                torch.cat(img_grid, dim=0),
                join(output_dir, f"grid_t{temperature:.3f}_eta{ddim_eta:.3f}.jpg"),
                nrow=num_images,
            )


def main():
    args = parse_arguments()

    embedded_prompts_dir = args.embedded_prompts_dir
    output_dir = args.output_dir
    num_images = args.num_images
    batch_size = args.batch_size
    temperature = args.temperature
    ddim_eta = args.ddim_eta
    clear_output_dir = args.clear_output_dir
    cuda_device = get_device(args.cuda_device)
    low_vram = args.low_vram
    seed = args.seed
    sampler = args.sampler
    cfg_scale = args.cfg_scale

    # override the batch_size if low_vram flag is set
    if low_vram:
        batch_size = 1

    seed_array = get_seed_array_from_string(seed, array_size=(num_images))

    generate_images_from_embeddings(
        embeddings_dir=embedded_prompts_dir,
        output_dir=output_dir,
        num_images= num_images,
        temperature=temperature,
        batch_size=batch_size,
        seed_array=seed_array,
        clear_output_dir=clear_output_dir,
        ddim_eta=ddim_eta,
        cuda_device=cuda_device,
        sampler_name=sampler,
        cfg_scale=cfg_scale
    )


if __name__ == "__main__":
    main()
