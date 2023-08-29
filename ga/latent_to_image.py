import os
import sys
import time

base_dir = os.getcwd()
sys.path.insert(0, base_dir)

import random
from os.path import join

import argparse
import csv

# import safetensors as st

from configs.model_config import ModelPathConfig
from stable_diffusion import StableDiffusion, SDconfigs
# TODO: rename stable_diffusion.utils_backend to /utils/cuda.py
from stable_diffusion.utils_backend import get_device
from stable_diffusion.utils_image import *


N_STEPS = 20  # 20, 12
CFG_STRENGTH = 9
DEVICE = get_device()

config = ModelPathConfig()


# Load Stable Diffusion
sd = StableDiffusion(device=DEVICE, n_steps=N_STEPS)
sd.quick_initialize().load_autoencoder(config.get_model(SDconfigs.VAE)).load_decoder(config.get_model(SDconfigs.VAE_DECODER))
sd.model.load_unet(config.get_model(SDconfigs.UNET))


latent = sd.generate_images(
    seed=SEED,
    prompt=prompt,
    null_prompt=NULL_PROMPT,
    uncond_scale=CFG_STRENGTH
    )