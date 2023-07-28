import os
import sys
import configparser
import argparse
import json


config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
parser = argparse.ArgumentParser(description="Setup a config file with the default IO directory structure.")

parser.add_argument("--base_io_directory_prefix", type=str, default="io/")
parser.add_argument("--base_directory", type=str, default="./")
parser.add_argument("--root_models_prefix", type=str, default="input/model/")
parser.add_argument("--root_outputs_prefix", type=str, default="output/model/")
parser.add_argument("--model_name", type=str, default="v1-5-pruned-emaonly")
parser.add_argument("--clip_model_name", type=str, default="vit-large-patch14")
args = parser.parse_args()

BASE_DIRECTORY = args.base_directory
BASE_IO_DIRECTORY_PREFIX = args.base_io_directory_prefix
ROOT_MODELS_PREFIX = args.root_models_prefix
ROOT_OUTPUTS_PREFIX = args.root_outputs_prefix
MODEL_NAME = args.model_name
CLIP_MODEL_NAME = args.clip_model_name
CHECKPOINT = f"{MODEL_NAME}.safetensors"
BASE_IO_DIRECTORY = os.path.join(BASE_DIRECTORY, BASE_IO_DIRECTORY_PREFIX)
print_section = lambda config, section: print(f"config.ini [{section}]: ", json.dumps({k: v for k, v in config[section].items()}, indent=4))

config["BASE"] = {
                    "BASE_DIRECTORY": f"{BASE_DIRECTORY}",
                    "BASE_IO_DIRECTORY": "${BASE_DIRECTORY}${BASE_IO_DIRECTORY_PREFIX}",
                    "BASE_IO_DIRECTORY_PREFIX": f"{BASE_IO_DIRECTORY_PREFIX}",
                    "ROOT_MODELS_PREFIX": f"{ROOT_MODELS_PREFIX}",
                    "ROOT_OUTPUTS_PREFIX": f"{ROOT_OUTPUTS_PREFIX}",
                    "MODEL_NAME": f"{MODEL_NAME}",
                    "CLIP_MODEL_NAME": f"{CLIP_MODEL_NAME}",
                    "CHECKPOINT": CHECKPOINT
                    }

print("config.ini [BASE]: ", json.dumps({k: v for k, v in config["BASE"].items()}, indent=4))

ROOT_MODELS_DIR = (os.path.join(BASE_IO_DIRECTORY, ROOT_MODELS_PREFIX))
ROOT_OUTPUTS_DIR = (os.path.join(BASE_IO_DIRECTORY, ROOT_OUTPUTS_PREFIX))
SD_DEFAULT_MODEL_OUTPUTS_DIR = (os.path.join(ROOT_OUTPUTS_PREFIX, MODEL_NAME))
SD_DEFAULT_MODEL_DIR = os.path.join(ROOT_MODELS_DIR, MODEL_NAME)
CLIP_MODELS_DIR = os.path.join(ROOT_MODELS_DIR, "clip")
TEXT_EMBEDDER_DIR = (
    os.path.join(CLIP_MODELS_DIR, "text_embedder")
)
IMAGE_ENCODER_DIR = (
    os.path.join(CLIP_MODELS_DIR, "image_encoder")
)

config["ROOT_DIRS"] = {
        'ROOT_MODELS_DIR':  '${BASE:base_io_directory}/${BASE:root_models_prefix}',
        'ROOT_OUTPUTS_DIR':  '${BASE:base_io_directory}/${BASE:root_outputs_prefix}',
    }

print_section(config, "ROOT_DIRS")
config["MODELS_DIRS"] = {
        'SD_DEFAULT_MODEL_DIR':  '${ROOT_DIRS:ROOT_MODELS_DIR}${BASE:MODEL_NAME}',
        'CLIP_MODELS_DIR':  '${ROOT_DIRS:ROOT_MODELS_DIR}clip',
}
# config["MODELS_DIRS"] = dict(
#         SD_DEFAULT_MODEL_DIR = SD_DEFAULT_MODEL_DIR,
#         CLIP_MODELS_DIR = CLIP_MODELS_DIR,
#         )

print_section(config, "MODELS_DIRS")
config["SUBMODELS_DIRS"] = {
        'TEXT_EMBEDDER_DIR': '${MODELS_DIRS:CLIP_MODELS_DIR}/text_embedder/',
        'IMAGE_ENCODER_DIR': '${MODELS_DIRS:CLIP_MODELS_DIR}/image_encoder/'
}

# config["SUBMODELS_DIRS"] = dict(
#         TEXT_EMBEDDER_DIR = TEXT_EMBEDDER_DIR,
#         IMAGE_ENCODER_DIR = IMAGE_ENCODER_DIR
# )
print_section(config, "SUBMODELS_DIRS")

config["STABLE_DIFFUSION_PATHS"] = dict(
    
    CHECKPOINT_PATH = os.path.join(ROOT_MODELS_DIR, CHECKPOINT),
    
    TEXT_EMBEDDER_PATH = (
        os.path.join(CLIP_MODELS_DIR, "text_embedder.safetensors")
    ),
    UNET_PATH = (
        os.path.join(SD_DEFAULT_MODEL_DIR, "unet.safetensors")
    ),
    AUTOENCODER_PATH = (
        os.path.join(SD_DEFAULT_MODEL_DIR, "autoencoder.safetensors")
    ),
    LATENT_DIFFUSION_PATH = (
        os.path.join(SD_DEFAULT_MODEL_DIR, "latent_diffusion.safetensors")
    )
)
print_section(config, "STABLE_DIFFUSION_PATHS")

config["CLIP_PATHS"] = dict(
    IMAGE_PROCESSOR_PATH = (
        os.path.join(IMAGE_ENCODER_DIR, "image_processor.ckpt")
    ),
    CLIP_MODEL_PATH = (
        os.path.join(IMAGE_ENCODER_DIR, "clip_model.ckpt")
    ),
    IMAGE_ENCODER_PATH = (
        os.path.join(IMAGE_ENCODER_DIR, "clip_image_encoder.ckpt")
    ),
    TOKENIZER_PATH = (
        os.path.join(TEXT_EMBEDDER_DIR, "tokenizer")
    ),
    TEXT_MODEL_PATH = (
        os.path.join(TEXT_EMBEDDER_DIR, CLIP_MODEL_NAME)
    ),
    )
print_section(config, "CLIP_PATHS")
with open('config.ini', 'w') as configfile:
    config.write(configfile)