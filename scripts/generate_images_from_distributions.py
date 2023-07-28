import os
import torch
import shutil
import argparse
import sys

base_directory = "./"
sys.path.insert(0, base_directory)

from tqdm import tqdm

from auxiliary_functions import get_torch_distribution_from_name

# from text_to_image import Txt2Img

from stable_diffusion.stable_diffusion import StableDiffusion
from stable_diffusion.constants import CHECKPOINT_PATH
from utility.labml.monit import section
from stable_diffusion.utils.utils import save_image_grid, save_images, get_device

OUTPUT_DIR = os.path.abspath("./output/noise-tests/from_distributions")

NOISE_SEEDS = [2982, 4801, 1995, 3598, 987, 3688, 8872, 762]

# CHECKPOINT_PATH = os.path.abspath('./input/model/v1-5-pruned-emaonly.ckpt')

parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "-p",
    "--prompt",
    type=str,
    default="A woman with flowers in her hair in a courtyard, in the style of Frank Frazetta",
    help="The prompt to generate images from. Defaults to 'A woman with flowers in her hair in a courtyard, in the style of Frank Frazetta'",
)
parser.add_argument(
    "-od",
    "--output_dir",
    type=str,
    default=OUTPUT_DIR,
    help="The output directory. defaults to OUTPUT_DIR constant, which should be './output/noise-tests/from_distributions'",
)
parser.add_argument(
    "-cp",
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

parser.add_argument("-bs", "--batch_size", type=int, default=1)
parser.add_argument("--params_steps", type=int, default=3)
parser.add_argument("--params_range", nargs="+", type=float, default=[0.49, 0.54])
parser.add_argument("--num_seeds", type=int, default=3)
parser.add_argument("-t", "--temperature", type=float, default=1.0)
parser.add_argument("--ddim_eta", type=float, default=0.0)
parser.add_argument("--clear_output_dir", type=bool, default=False)
parser.add_argument("--cuda_device", type=str, default=get_device())

args = parser.parse_args()

PROMPT = args.prompt
OUTPUT_DIR = args.output_dir
CHECKPOINT_PATH = args.checkpoint_path
FULLY_INIT = args.fully_initialize
DISTRIBUTION_ID = args.distribution_index
NUM_SEEDS = args.num_seeds
BATCH_SIZE = args.batch_size
PARAMS_STEPS = args.params_steps
PARAMS_RANGE = args.params_range
TEMPERATURE = args.temperature
DDIM_ETA = args.ddim_eta
CLEAR_OUTPUT_DIR = args.clear_output_dir
DEVICE = args.cuda_device

NOISE_SEEDS = NOISE_SEEDS[:NUM_SEEDS]

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


def init_stable_diffusion(device, sampler_name="ddim", n_steps=20, ddim_eta=0.0):
    device = get_device(device)

    stable_diffusion = StableDiffusion(
        device=device, sampler_name=sampler_name, n_steps=n_steps, ddim_eta=ddim_eta
    )

    if not FULLY_INIT:
        stable_diffusion.quick_initialize().load_submodel_tree()
        return stable_diffusion

    else:
        stable_diffusion.initialize_latent_diffusion(
            path=CHECKPOINT_PATH, force_submodels_init=True
        )
        return stable_diffusion


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


def generate_images_from_dist_dict(
    stable_diffusion: StableDiffusion,
    distributions: dict,
    output_dir: str = OUTPUT_DIR,
    clear_output_dir: bool = CLEAR_OUTPUT_DIR,
    prompt: str = PROMPT,
    batch_size: int = BATCH_SIZE,
    temperature: float = TEMPERATURE,
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
    total_images = len(NOISE_SEEDS) * len(VAR_RANGE)
    # Generate the images
    img_grids = []
    img_counter = 0

    for distribution_index, (distribution_name, params) in enumerate(
        distributions.items()
    ):
        
        with torch.no_grad():
            with tqdm(
                total=total_images,
                desc="Generating images",
            ) as pbar:
                print(f"Generating images for {distribution_name}")
                noise_fn = (
                    lambda shape, device=None: get_torch_distribution_from_name(DIST_NAME)(
                        **params
                    )
                    .sample(shape)
                    .to(device)
                )

                grid_rows = []

                for seed_index, noise_seed in enumerate(NOISE_SEEDS):
                    p_bar_description = f"Generating image {img_counter+1} of {total_images}. Distribution: {DIST_NAME}{params}"
                    pbar.set_description(p_bar_description)

                    # image_name = f"n{noise_seed:04d}_d{distribution_name}p{'_'.join(list(params.values())):04d}.jpg"
                    image_name = f"n{noise_seed:04d}_d{distribution_name}.jpg"
                    dest_path = os.path.join(
                        os.path.join(output_dir, distribution_name), image_name
                    )
                    images = stable_diffusion.generate_images(
                        batch_size=batch_size,
                        prompt=prompt,
                        seed=noise_seed,
                        noise_fn=noise_fn,
                        temperature=temperature,
                    )
                    grid_rows.append(images)
                    save_images(images, dest_path=dest_path)
                    img_counter += 1
                    pbar.update(1)

                dest_path = os.path.join(
                    os.path.join(output_dir, distribution_name), f"grid_{distribution_name}.jpg"
                )
                grid = torch.cat(grid_rows, dim=0)
        img_grids.append(grid)

    dest_path = os.path.join(
        output_dir,
        f"grid_all_{DIST_NAME}{VAR_RANGE[0].item():.2f}_{VAR_RANGE[-1].item():.2f}.jpg",
    )
    grid = torch.cat(img_grids, dim=0)
    # torch.save(grid, dest_path.replace(".jpg", ".pt"))
    # st.save_file({"img_grid": grid}, dest_path.replace('.jpg', '.safetensors'))
    save_image_grid(grid, dest_path, nrow=PARAMS_STEPS, normalize=True, scale_each=True)


def main():
    sd = init_stable_diffusion(DEVICE, ddim_eta=DDIM_ETA)

    generate_images_from_dist_dict(
        sd,
        DISTRIBUTIONS,
        output_dir=OUTPUT_DIR,
        clear_output_dir=CLEAR_OUTPUT_DIR,
        prompt=PROMPT,
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
    )


if __name__ == "__main__":
    main()
