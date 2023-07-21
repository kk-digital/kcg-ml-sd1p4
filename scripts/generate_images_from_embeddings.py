import os
import torch
import shutil
import argparse
import sys

base_directory = "./"
sys.path.insert(0, base_directory)


from tqdm import tqdm
from stable_diffusion.constants import (
    CHECKPOINT_PATH,
    AUTOENCODER_PATH,
    UNET_PATH,
    EMBEDDER_PATH,
    LATENT_DIFFUSION_PATH,
    ENCODER_PATH,
    DECODER_PATH,
    TOKENIZER_PATH,
    TRANSFORMER_PATH,
)
from stable_diffusion.stable_diffusion import StableDiffusion
from labml.monit import section
from stable_diffusion.utils.utils import save_image_grid, save_images, check_device

from os.path import join

# CHECKPOINT_PATH = os.path.abspath('./input/model/v1-5-pruned-emaonly.ckpt')

OUTPUT_DIR = os.path.abspath("./output/noise-tests/from_embeddings")
EMBEDDED_PROMPTS_DIR = os.path.abspath("./input/embedded_prompts/")

NOISE_SEEDS = [2982, 4801, 1995, 3598, 987, 3688, 8872, 762]

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
parser.add_argument("--num_seeds", type=int, default=3)
parser.add_argument("-bs", "--batch_size", type=int, default=1)
parser.add_argument("-t", "--temperature", type=float, default=1.0)
parser.add_argument("--ddim_eta", type=float, default=0.0)
parser.add_argument("--clear_output_dir", type=bool, default=False)
parser.add_argument("--cuda_device", type=str, default="cuda:0")

args = parser.parse_args()

EMBEDDED_PROMPTS_DIR = args.embedded_prompts_dir
OUTPUT_DIR = args.output_dir
NUM_SEEDS = args.num_seeds
BATCH_SIZE = args.batch_size
TEMPERATURE = args.temperature
DDIM_ETA = args.ddim_eta
CLEAR_OUTPUT_DIR = args.clear_output_dir
DEVICE = args.cuda_device

NOISE_SEEDS = NOISE_SEEDS[:NUM_SEEDS]


def init_stable_diffusion(device, sampler_name="ddim", n_steps=20, ddim_eta=0.0):
    device = check_device(device)

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
    batch_size: int = BATCH_SIZE,
    noise_seeds: list = NOISE_SEEDS,
    clear_output_dir: bool = CLEAR_OUTPUT_DIR,
    ddim_eta: float = DDIM_ETA,
    cuda_device: str = DEVICE,
):
    null_cond = torch.load(join(embeddings_dir, "null_cond.pt"), map_location=DEVICE)
    # print(null_cond.shape)
    embeddings = torch.load(
        join(embeddings_dir, "embedded_prompts.pt"), map_location=DEVICE
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

    total_images = len(noise_seeds) * len(embeddings)

    with torch.no_grad():
        with tqdm(
            total=total_images,
            desc="Generating images",
        ) as pbar:
            img_count = 0
            img_grid = []
            for prompt_index, prompt in enumerate(embeddings):
                img_row = []
                for seed_index, noise_seed in enumerate(noise_seeds):
                    p_bar_description = (
                        f"Generating image {img_count+1} of {total_images}"
                    )
                    pbar.set_description(p_bar_description)

                    image_name = f"n{noise_seed:04d}_a{prompt_index+1:04d}.jpg"
                    dest_path = join(output_dir, image_name)

                    images = stable_diffusion.generate_images_from_embeddings(
                        batch_size=batch_size,
                        embedded_prompt=prompt.unsqueeze(0),
                        null_prompt=null_cond,
                        seed=noise_seed,
                        temperature=TEMPERATURE,
                    )
                    # print(images.shape)
                    save_images(images, dest_path=dest_path)
                    img_row.append(images)
                    img_count += 1
                    pbar.update(1)
                img_grid.append(torch.cat(img_row, dim=0))
            save_image_grid(
                torch.cat(img_grid, dim=0),
                join(output_dir, f"grid_t{TEMPERATURE:.3f}_eta{DDIM_ETA:.3f}.jpg"),
                nrow=NUM_SEEDS,
            )


def main():
    # args = CLI('Generate images from noise seeds.') \
    #     .prompt_prefix() \
    #     .artist_file() \
    #     .output() \
    #     .checkpoint_path() \
    #     .sampler() \
    #     .steps() \
    #     .batch_size() \
    #     .parse()

    # generate_images(
    #     prompt_prefix=args.prompt_prefix,
    #     artist_file=args.artist_file,
    #     output_dir=args.output,
    #     checkpoint_path=args.checkpoint_path,
    #     sampler_name=args.sampler,
    #     n_steps=args.steps,
    # )
    # artists_file = os.path.abspath('./input/artists.txt')
    # generate_images(artist_file=artists_file)
    generate_images_from_embeddings(
        embeddings_dir=EMBEDDED_PROMPTS_DIR,
        output_dir=OUTPUT_DIR,
        # sampler_name="ddim",
        # n_steps=20,
        batch_size=BATCH_SIZE,
        noise_seeds=NOISE_SEEDS,
        clear_output_dir=CLEAR_OUTPUT_DIR,
        ddim_eta=DDIM_ETA,
        cuda_device=DEVICE,
    )


if __name__ == "__main__":
    main()
