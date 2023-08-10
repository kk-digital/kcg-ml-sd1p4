import argparse
import os
import shutil
import sys
from random import randrange
import torch

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from stable_diffusion.utils_backend import get_device
from auxiliary_functions import get_torch_distribution_from_name
from stable_diffusion.utils_image import save_image_grid, save_images
from stable_diffusion.constants import CHECKPOINT_PATH
from stable_diffusion import StableDiffusion
from utility.labml.monit import section

# CHECKPOINT_PATH = os.path.abspath('./input/model/v1-5-pruned-emaonly.ckpt')
OUTPUT_DIR = os.path.abspath("./output/noise-tests/temperature_range")

parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "--prompt",
    type=str,
    default="A woman with flowers in her hair in a courtyard, in the style of Frank Frazetta",
    help="The prompt to generate images from. Defaults to 'A woman with flowers in her hair in a courtyard, in the style of Frank Frazetta'",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=OUTPUT_DIR,
    help="The output directory. defaults to OUTPUT_DIR constant, which should be './output/noise-tests/temperature_range'",
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default=CHECKPOINT_PATH,
    help="The path to the checkpoint file to load from. Defaults to CHECKPOINT_PATH constant, which should be './input/model/v1-5-pruned-emaonly.ckpt'",
)
parser.add_argument("-F", "--fully_initialize", type=bool, default=False)
parser.add_argument(
    "-d",
    "--distribution_index",
    type=int,
    default=4,
    help="0: 'Normal' | 1: 'Cauchy' | 2: 'Gumbel' | 3: 'Laplace' | 4: 'Logistic' | Defaults to 4.",
)
parser.add_argument("--seed", type=str, default='')
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--params_steps", type=int, default=3)
parser.add_argument("--params_range", nargs="+", type=float, default=[0.49, 0.54])
parser.add_argument("--temperature_steps", type=int, default=3)
parser.add_argument("--temperature_range", nargs="+", type=float, default=[1.0, 4.0])
parser.add_argument("--ddim_eta", type=float, default=0.1)
parser.add_argument("--clear_output_dir", type=bool, default=False)
parser.add_argument("--cuda_device", type=str, default=None)

args = parser.parse_args()

PROMPT = args.prompt
OUTPUT_DIR = args.output_dir
CHECKPOINT_PATH = args.checkpoint_path
FULLY_INIT = args.fully_initialize
DISTRIBUTION_ID = args.distribution_index

if args.seed == '':
    NOISE_SEED = randrange(0, 2 ** 24)
else:
    NOISE_SEED = int(args.seed)

BATCH_SIZE = args.batch_size
PARAMS_STEPS = args.params_steps
PARAMS_RANGE = args.params_range
TEMPERATURE_STEPS = args.temperature_steps
TEMPERATURE_RANGE = args.temperature_range
DDIM_ETA = args.ddim_eta
CLEAR_OUTPUT_DIR = args.clear_output_dir
DEVICE = get_device(args.cuda_device)

TEMP_RANGE = torch.linspace(*TEMPERATURE_RANGE, TEMPERATURE_STEPS)

__DISTRIBUTIONS = {
    "Normal": dict(loc=0.0, scale=1.0),
    "Cauchy": dict(loc=0.0, scale=1.0),
    "Gumbel": dict(loc=1.0, scale=2.0),
    "Laplace": dict(
        loc=0.0, scale=1.0
    ),  # there's some stuff here for scale \in (0.6, 0.8)
    "Logistic": dict(loc=0.0, scale=1.0),
}
dist_names = list(__DISTRIBUTIONS.keys())
DIST_NAME = dist_names[DISTRIBUTION_ID]
VAR_RANGE = torch.linspace(*PARAMS_RANGE, PARAMS_STEPS)
DISTRIBUTIONS = {
    f"{DIST_NAME}_{var.item():.4f}": dict(loc=0.0, scale=var.item())
    for var in VAR_RANGE
}


def create_folder_structure(
        distributions_dict: dict, root_dir: str = OUTPUT_DIR
) -> None:
    for i, distribution_name in enumerate(distributions_dict.keys()):
        distribution_outputs = os.path.join(root_dir, distribution_name)

        try:
            os.makedirs(distribution_outputs, exist_ok=True)
        except Exception as e:
            print(e)


def init_stable_diffusion(
        checkpoint_path: str = CHECKPOINT_PATH,
        sampler_name: str = "ddim",
        ddim_steps: int = 20,
        ddim_eta: float = 0.0,
):
    sd = StableDiffusion(
        device=DEVICE,
        sampler_name=sampler_name,
        ddim_steps=ddim_steps,
        ddim_eta=ddim_eta,
    )
    # compute loading time

    if not FULLY_INIT:
        with section("to initialize latent diffusion and load submodels tree"):
            sd.quick_initialize().load_submodel_tree()
            # latent_diffusion_model = initialize_latent_diffusion()
            # latent_diffusion_model.load_submodel_tree()
            # sd.initialize_from_model(latent_diffusion_model)
        return sd
    else:
        with section("to run `StableDiffusionBaseScript`'s initialization function"):
            sd.initialize_latent_diffusion(
                path=checkpoint_path,
                force_submodels_init=True,
            )

        return sd


def show_summary(total_time, partial_time, total_images, output_dir):
    print("[SUMMARY]")
    print("Total time taken: %.2f seconds" % total_time)
    print("Partial time (without initialization): %.2f seconds" % partial_time)
    print("Total images generated: %s" % total_images)
    print("Images/second: %.2f" % (total_images / total_time))
    print(
        "Images/second (without initialization): %.2f" % (total_images / partial_time)
    )

    print("Images generated successfully at", output_dir)


def generate_images_from_temp_range(
        distributions: dict,
        txt2img: StableDiffusion,
        output_dir: str = OUTPUT_DIR,
        clear_output_dir: bool = CLEAR_OUTPUT_DIR,
        prompt: str = PROMPT,
        noise_seed: int = NOISE_SEED,
        batch_size: int = BATCH_SIZE,
        temperature_range=TEMP_RANGE,
):
    # Clear the output directory

    if clear_output_dir:
        try:
            shutil.rmtree(output_dir)
        except Exception as e:
            print(e)

    os.makedirs(output_dir, exist_ok=True)
    # Create the folder structure for the outputs
    create_folder_structure(distributions, output_dir)

    # Generate the images
    img_grids = []
    for distribution_index, (distribution_name, params) in enumerate(
            distributions.items()
    ):
        noise_fn = (
            lambda shape, device=None: get_torch_distribution_from_name(DIST_NAME)(
                **params
            )
            .sample(shape)
            .to(device)
        )
        img_rows = []
        for temperature in temperature_range:
            images = txt2img.generate_images(
                batch_size=batch_size,
                prompt=prompt,
                seed=noise_seed,
                noise_fn=noise_fn,
                temperature=temperature.item(),
            )

            image_name = f"n{noise_seed:04d}_d{DIST_NAME}_t{temperature:.3f}_eta{DDIM_ETA:.3f}.jpg"
            dest_path = os.path.join(
                os.path.join(output_dir, distribution_name), image_name
            )

            img_rows.append(images)
            save_images(images, dest_path=dest_path)

        row = torch.cat(img_rows, dim=0)
        img_grids.append(row)

    dest_path = os.path.join(
        output_dir,
        f"t{TEMPERATURE_RANGE[0]:.3f}t{TEMPERATURE_RANGE[1]:.3f}_vs_{DIST_NAME}_p{PARAMS_RANGE[0]:.3f}p{PARAMS_RANGE[1]:.3f}.jpg",
    )

    grid = torch.cat(img_grids, dim=0)
    save_image_grid(grid, dest_path, nrow=PARAMS_STEPS, normalize=True, scale_each=True)


def main():
    txt2img = init_stable_diffusion(ddim_eta=DDIM_ETA)
    generate_images_from_temp_range(
        DISTRIBUTIONS,
        txt2img,
        output_dir=OUTPUT_DIR,
        clear_output_dir=CLEAR_OUTPUT_DIR,
        prompt=PROMPT,
        noise_seed=NOISE_SEED,
        batch_size=BATCH_SIZE,
        temperature_range=TEMP_RANGE,
    )


if __name__ == "__main__":
    main()

# %%
