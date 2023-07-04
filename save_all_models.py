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

import random
import os
from pathlib import Path
from typing import Union

import PIL
import numpy as np
import torch
from PIL import Image

# from labml import monit
# from labml.logger import inspect


from stable_diffusion2.constants import AUTOENCODER_PATH, ENCODER_PATH, DECODER_PATH
from stable_diffusion2.constants import EMBEDDER_PATH, TOKENIZER_PATH, TRANSFORMER_PATH
from stable_diffusion2.constants import UNET_PATH
from stable_diffusion2.constants import LATENT_DIFFUSION_PATH
from stable_diffusion2.constants import CHECKPOINT_PATH

from stable_diffusion2.utils.model import initialize_autoencoder, initialize_clip_embedder, initialize_unet, initialize_latent_diffusion
from stable_diffusion2.utils.utils import SectionManager as section

from stable_diffusion2.latent_diffusion import LatentDiffusion
from stable_diffusion2.model.vae.encoder import Encoder
from stable_diffusion2.model.vae.decoder import Decoder
from stable_diffusion2.model.vae.autoencoder import Autoencoder
from stable_diffusion2.model.clip.clip_embedder import CLIPTextEmbedder
from stable_diffusion2.model.unet.unet import UNetModel
# from stable_diffusion2.model.unet import UNetModel
from torchinfo import summary

if __name__ == '__main__':
    assert len(os.sys.argv) > 1, 'Please provide an argument.'
    if int(os.sys.argv[1]) == 1:
        embedder = initialize_clip_embedder()
        embedder.save_submodels()
        embedder.save()
        summary(embedder)
    elif int(os.sys.argv[1]) == 2:
        autoencoder = initialize_autoencoder()
        autoencoder.save_submodels()
        autoencoder.save()
        summary(autoencoder)
    elif int(os.sys.argv[1]) == 3:
        unet = initialize_unet()
        unet.save()
        summary(unet)
    elif int(os.sys.argv[1]) == 0:
        
        embedder = initialize_clip_embedder()
        embedder.save_submodels()
        embedder.save()
        
        autoencoder = initialize_autoencoder()
        autoencoder.save_submodels()
        autoencoder.save()
        
        unet = initialize_unet()
        unet.save()

        summary(embedder)
        summary(autoencoder)
        summary(unet)

    else:
        if len(os.sys.argv) > 2:
            if os.sys.argv[2] == 'True':
                model = initialize_latent_diffusion(path=CHECKPOINT_PATH, force_submodels_init=True)
                summary(model)
        else:
            model = initialize_latent_diffusion(path=CHECKPOINT_PATH, force_submodels_init=False)
        model.save()
        print(type(model))