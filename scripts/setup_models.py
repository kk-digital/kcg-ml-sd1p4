"""
---
title: Utility functions for stable diffusion
summary: >
 Utility functions for stable diffusion
---

# initializes and saves all models
1 saves clip embedder
2 saves autoencoder
3 saves unet
0 saves all submodels
else saves latent diffusion model with state dict loaded from checkpoint
"""

import sys
import argparse
import configparser

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.labml.monit import section
from stable_diffusion.constants import CLIP_MODEL_DIR
from stable_diffusion.utils_model import (
    initialize_latent_diffusion,
)
from stable_diffusion.model.clip_image_encoder import CLIPImageEncoder


try:
    from torchinfo import summary
except:
    print("torchinfo not installed")
    summary = lambda x: print(x)

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read("./config.ini")

base = config["BASE"]
root_dirs = config["ROOT_DIRS"]
sd_paths = config['STABLE_DIFFUSION_PATHS']
CHECKPOINT_PATH = sd_paths.get('checkpoint_path')
ROOT_MODELS_DIR = root_dirs.get('root_models_dir')

parser = argparse.ArgumentParser(description="")
parser.add_argument("--root_models_dir", type=str, default=ROOT_MODELS_DIR)
parser.add_argument("--checkpoint_path", type=str, default=CHECKPOINT_PATH)
# parser.add_argument(
#     "--without_weights",
#     default=False,
#     action="store_true",
#     help="Save the submodels without loading weights from checkpoint",
# )
# parser.add_argument("--unet", default=False, action="store_true")
# parser.add_argument(
#     "--clip",
#     default=False,
#     action="store_true",
# )
# parser.add_argument("--vae", default=False, action="store_true")
# parser.add_argument("--image_encoder", type=bool, default=True)

args = parser.parse_args()


# print(args)
CHECKPOINT_PATH = args.checkpoint_path
ROOT_MODELS_DIR = args.root_models_dir





if __name__ == "__main__":

    model = initialize_latent_diffusion(
            path=CHECKPOINT_PATH,
            force_submodels_init=True
        )
    summary(model)

    with section(
        "initialize CLIP image encoder and load submodels from lib"
    ):
        img_encoder = CLIPImageEncoder()
        img_encoder.load_submodels(image_processor_path=CLIP_MODEL_DIR, vision_model_path=CLIP_MODEL_DIR)
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
