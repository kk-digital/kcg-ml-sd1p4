# %%
import os
import sys
import torch
import time

base_directory = "../"
sys.path.insert(0, base_directory)

from stable_diffusion2.latent_diffusion import LatentDiffusion
from stable_diffusion2.stable_diffusion import StableDiffusion
from stable_diffusion2.utils.model import *
from stable_diffusion2.utils.utils import SectionManager as section
from stable_diffusion2.utils.utils import *
from stable_diffusion2.model.clip.clip_embedder import CLIPTextEmbedder



from stable_diffusion2.model.unet.unet import UNetModel

from pathlib import Path

device = get_device()

# %%
get_memory_status()

# %%
latent_diffusion_model = LatentDiffusion(linear_start=0.00085,
            linear_end=0.0120,
            n_steps=1000,
            latent_scaling_factor=0.18215
            )

# %%
get_memory_status()

# %%
latent_diffusion_model.load_autoencoder()
# latent_diffusion_model.load_submodel_tree()

# %%
get_memory_status()

# %%
latent_diffusion_model.first_stage_model.load_encoder()

# %%
get_memory_status()

# %%
img = load_img("./scripts2/test_img.jpg").to(device)

# %%
encoded_img = latent_diffusion_model.autoencoder_encode(img)

# %%
print(get_memory_status())

# %%
latent_diffusion_model.first_stage_model.unload_decoder()

# %%
get_memory_status()

# %%
latent_diffusion_model.first_stage_model.load_decoder()

# %%
get_memory_status()

# %%
decoded_img = latent_diffusion_model.autoencoder_decode(encoded_img)
