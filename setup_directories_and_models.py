import os
import configparser
import argparse
import json
import requests

from stable_diffusion.utils_logger import logger
from utility.labml.monit import section
from stable_diffusion.utils_model import initialize_latent_diffusion
from stable_diffusion.model.clip_image_encoder import CLIPImageEncoder

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
parser = argparse.ArgumentParser(description="Setup a config file with the default IO directory structure.")

parser.add_argument("--base_io_directory_prefix", type=str, default="")
parser.add_argument("--base_directory", type=str, default="./")
parser.add_argument("--root_models_prefix", type=str, default="input/model/")
parser.add_argument("--root_outputs_prefix", type=str, default="output/model/")
parser.add_argument("--model_name", type=str, default="v1-5-pruned-emaonly")
parser.add_argument("--clip_model_name", type=str, default="vit-large-patch14")
parser.add_argument("--download_base_clip_model", type=bool, default=False)
parser.add_argument("--download_base_sd_model", type=bool, default=False)
args = parser.parse_args()

BASE_DIRECTORY = args.base_directory
BASE_IO_DIRECTORY_PREFIX = args.base_io_directory_prefix
ROOT_MODELS_PREFIX = args.root_models_prefix
ROOT_OUTPUTS_PREFIX = args.root_outputs_prefix
MODEL_NAME = args.model_name
CLIP_MODEL_NAME = args.clip_model_name
DOWNLOAD_BASE_CLIP_MODEL = args.download_base_clip_model
DOWNLOAD_BASE_SD_MODEL = args.download_base_sd_model
CHECKPOINT = f"{MODEL_NAME}.safetensors"
BASE_IO_DIRECTORY = os.path.join(BASE_DIRECTORY, BASE_IO_DIRECTORY_PREFIX)

print_section = lambda config, section: print(f"config.ini [{section}]: ",
                                              json.dumps({k: v for k, v in config[section].items()}, indent=4))

config["BASE"] = {
    "BASE_DIRECTORY": f"{BASE_DIRECTORY}",
    "BASE_IO_DIRECTORY": "${BASE_DIRECTORY}${BASE_IO_DIRECTORY_PREFIX}",
    "BASE_IO_DIRECTORY_PREFIX": f"{BASE_IO_DIRECTORY_PREFIX}",
    "ROOT_MODELS_PREFIX": f"{ROOT_MODELS_PREFIX}",
    "ROOT_OUTPUTS_PREFIX": f"{ROOT_OUTPUTS_PREFIX}",
    "MODEL_NAME": f"{MODEL_NAME}",
    "CLIP_MODEL_NAME": f"{CLIP_MODEL_NAME}",
    "CHECKPOINT": "${MODEL_NAME}.safetensors"
}

print_section(config, 'BASE')

# ROOT_MODELS_DIR = (os.path.join(BASE_IO_DIRECTORY, ROOT_MODELS_PREFIX))
# ROOT_OUTPUTS_DIR = (os.path.join(BASE_IO_DIRECTORY, ROOT_OUTPUTS_PREFIX))
# SD_DEFAULT_MODEL_OUTPUTS_DIR = (os.path.join(ROOT_OUTPUTS_PREFIX, MODEL_NAME))
# SD_DEFAULT_MODEL_DIR = os.path.join(ROOT_MODELS_DIR, MODEL_NAME)
# CLIP_MODELS_DIR = os.path.join(ROOT_MODELS_DIR, "clip/")
# TEXT_EMBEDDER_DIR = (
#     os.path.join(CLIP_MODELS_DIR, "text_embedder/")
# )
# IMAGE_ENCODER_DIR = (
#     os.path.join(CLIP_MODELS_DIR, "image_encoder/")
# )

config["ROOT_DIRS"] = {
    'ROOT_MODELS_DIR': '${BASE:base_io_directory}${BASE:root_models_prefix}',
    'ROOT_OUTPUTS_DIR': '${BASE:base_io_directory}${BASE:root_outputs_prefix}',
}

print_section(config, "ROOT_DIRS")
config["MODELS_DIRS"] = {
    'SD_DEFAULT_MODEL_DIR': '${ROOT_DIRS:ROOT_MODELS_DIR}${BASE:MODEL_NAME}/',
    'CLIP_MODELS_DIR': '${ROOT_DIRS:ROOT_MODELS_DIR}clip/',
    'CLIP_MODEL_DIR': '${MODELS_DIRS:CLIP_MODELS_DIR}${BASE:CLIP_MODEL_NAME}/',
    'TEXT_EMBEDDER_DIR': '${MODELS_DIRS:CLIP_MODELS_DIR}text_embedder/',
    'IMAGE_ENCODER_DIR': '${MODELS_DIRS:CLIP_MODELS_DIR}image_encoder/'
}
# config["MODELS_DIRS"] = dict(
#         SD_DEFAULT_MODEL_DIR = SD_DEFAULT_MODEL_DIR,
#         CLIP_MODELS_DIR = CLIP_MODELS_DIR,
#         )

print_section(config, "MODELS_DIRS")
config["SUBMODELS_DIRS"] = {
    'TOKENIZER_DIR': '${MODELS_DIRS:TEXT_EMBEDDER_DIR}tokenizer/',
    'TEXT_MODEL_DIR': '${MODELS_DIRS:TEXT_EMBEDDER_DIR}text_model/',
    'IMAGE_PROCESSOR_DIR': '${MODELS_DIRS:IMAGE_ENCODER_DIR}image_processor/',
    'VISION_MODEL_DIR': '${MODELS_DIRS:IMAGE_ENCODER_DIR}vision_model/',
}

# config["SUBMODELS_DIRS"] = dict(
#         TEXT_EMBEDDER_DIR = TEXT_EMBEDDER_DIR,
#         IMAGE_ENCODER_DIR = IMAGE_ENCODER_DIR
# )
print_section(config, "SUBMODELS_DIRS")

config["STABLE_DIFFUSION_PATHS"] = {
    'CHECKPOINT_PATH': '${ROOT_DIRS:ROOT_MODELS_DIR}${BASE:CHECKPOINT}',
    'UNET_PATH': '${MODELS_DIRS:SD_DEFAULT_MODEL_DIR}unet.safetensors',
    'AUTOENCODER_PATH': "${MODELS_DIRS:SD_DEFAULT_MODEL_DIR}autoencoder.safetensors",
    'LATENT_DIFFUSION_PATH': "${MODELS_DIRS:SD_DEFAULT_MODEL_DIR}latent_diffusion.safetensors"
}
print_section(config, "STABLE_DIFFUSION_PATHS")

config["CLIP_PATHS"] = dict(

    IMAGE_PROCESSOR_PATH="${MODELS_DIRS:IMAGE_ENCODER_DIR}image_processor",
    VISION_MODEL_PATH="${MODELS_DIRS:IMAGE_ENCODER_DIR}vision_model",
    IMAGE_ENCODER_PATH="${MODELS_DIRS:IMAGE_ENCODER_DIR}image_encoder.safetensors",

    TOKENIZER_PATH="${MODELS_DIRS:TEXT_EMBEDDER_DIR}tokenizer",
    TEXT_MODEL_PATH='${MODELS_DIRS:TEXT_EMBEDDER_DIR}text_model',
    TEXT_EMBEDDER_PATH="${MODELS_DIRS:TEXT_EMBEDDER_DIR}text_embedder.safetensors"
)

print_section(config, "CLIP_PATHS")

with open('config.ini', 'w') as configfile:
    config.write(configfile)


def create_directory_tree_folders(config):
    for section in config.sections():
        if section.endswith("_DIRS"):
            for k, v in config[section].items():
                os.makedirs(v, exist_ok=True)


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def download_file(url, file_path, description):
    if not os.path.isfile(file_path):
        with section(f"Initializing {description} download."):
            r = requests.get(url, allow_redirects=True)
            try:
                with open(file_path, 'wb') as f:
                    f.write(r.content)
                print(f"{description} downloaded successfully.")
            except Exception as e:
                print(f"Error downloading {description}. Error: ", e)
    else:
        logger.info(f"{description} already exists.")


if __name__ == "__main__":

    create_directory_tree_folders(config)

    if DOWNLOAD_BASE_CLIP_MODEL:
        clip_model_dir = config["MODELS_DIRS"].get('clip_model_dir')
        create_directory_if_not_exists(clip_model_dir)
        clip_path = os.path.join(clip_model_dir, 'model.safetensors')
        clip_url = r'https://huggingface.co/openai/clip-vit-large-patch14/resolve/refs%2Fpr%2F19/model.safetensors'
        download_file(clip_url, clip_path, "CLIP model")

    if DOWNLOAD_BASE_SD_MODEL:
        root_models_dir = config["ROOT_DIRS"].get('root_models_dir')
        create_directory_if_not_exists(root_models_dir)
        sd_path = os.path.join(root_models_dir, 'v1-5-pruned-emaonly.safetensors')
        sd_url = r'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors'
        download_file(sd_url, sd_path, "Stable Diffusion checkpoint")

    model = initialize_latent_diffusion(
        path=config["STABLE_DIFFUSION_PATHS"].get('CHECKPOINT_PATH'),
        force_submodels_init=True
    )
    # summary(model)

    with section(
            "initialize CLIP image encoder and load submodels from lib"
    ):
        img_encoder = CLIPImageEncoder()
        img_encoder.load_submodels(image_processor_path=config["MODELS_DIRS"].get('CLIP_MODEL_DIR'),
                                   vision_model_path=config["MODELS_DIRS"].get('CLIP_MODEL_DIR'))
    with section("save image encoder submodels"):
        img_encoder.save_submodels()
        img_encoder.unload_submodels()
        img_encoder.save()
    with section("save vae submodels"):
        model.first_stage_model.save_submodels()  # saves autoencoder submodels (encoder, decoder) with loaded state dict
    with section("unload vae submodels"):
        model.first_stage_model.unload_submodels()  # unloads autoencoder submodels
    with section("save embedder submodels"):
        model.cond_stage_model.save_submodels()  # saves text embedder submodels (tokenizer, transformer) with loaded state dict
    with section("unload embedder submodels"):
        model.cond_stage_model.unload_submodels()  # unloads text embedder submodels
    with section("save latent diffusion submodels"):
        model.save_submodels()  # saves latent diffusion submodels (autoencoder, clip_embedder, unet) with loaded state dict and unloaded submodels (when it applies)
    with section("unload latent diffusion submodels"):
        model.unload_submodels()  # unloads latent diffusion submodels
    with section("save latent diffusion model"):
        model.save()  # saves latent diffusion model with loaded state dict and unloaded submodels
