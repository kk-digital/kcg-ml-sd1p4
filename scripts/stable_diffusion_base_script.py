import os
import sys
from pathlib import Path
from typing import Union, Optional

import torch
import time
sys.path.append(os.path.abspath(''))

from stable_diffusion.utils_backend import get_device
from stable_diffusion.utils_image import load_img
from stable_diffusion.sampler.ddim import DDIMSampler
from stable_diffusion.sampler.ddpm import DDPMSampler
from stable_diffusion.utils_model import initialize_latent_diffusion
from stable_diffusion.latent_diffusion import LatentDiffusion
from stable_diffusion.sampler.diffusion import DiffusionSampler
from stable_diffusion.model_paths import LATENT_DIFFUSION_PATH
from utility.labml.monit import section
from stable_diffusion.utils_backend import get_autocast, set_seed

class ModelLoadError(Exception):
    pass


class StableDiffusionBaseScript:
    model: LatentDiffusion
    sampler: DiffusionSampler

    def __init__(self, *, checkpoint_path: Union[str, Path] = None,
                 ddim_steps: int = 50,
                 ddim_eta: float = 0.0,
                 force_cpu: bool = False,
                 sampler_name: str = 'ddim',
                 n_steps: int = 20,
                 cuda_device: str = 'cuda:0',
                 ):
        """
        :param checkpoint_path: is the path of the checkpoint
        :param sampler_name: is the name of the [sampler](../sampler/index.html)
        :param n_steps: is the number of sampling steps
        :param ddim_eta: is the [DDIM sampling](../sampler/ddim.html) $\eta$ constant
        """
        self.checkpoint_path = checkpoint_path
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.force_cpu = force_cpu
        self.sampler_name = sampler_name
        self.n_steps = n_steps
        self.cuda_device = cuda_device
        self.device_id = get_device(cuda_device)
        self.device = torch.device(self.device_id)
        self.empty_embedding = None

        # Load [latent diffusion model](../latent_diffusion.html)
        # Get device or force CPU if requested

    def encode_image(self, orig_img: str, batch_size: int = 1):
        """
        Encode an image in the latent space
        """
        orig_image = load_img(orig_img).to(self.device)
        # Encode the image in the latent space and make `batch_size` copies of it
        orig = self.model.autoencoder_encode(orig_image).repeat(batch_size, 1, 1, 1)

        return orig

    def prepare_mask(self, mask: Optional[torch.Tensor], orig: torch.Tensor):
        # If `mask` is not provided,
        # we set a sample mask to preserve the bottom half of the image
        if mask is None:
            mask = torch.zeros_like(orig, device=self.device)
            mask[:, :, mask.shape[2] // 2:, :] = 1.
        else:
            mask = mask.to(self.device)

        return mask

    def calc_strength_time_step(self, strength: float):
        # Get the number of steps to diffuse the original
        t_index = int(strength * self.ddim_steps)

        return t_index

    def get_text_conditioning(self, uncond_scale: float, prompts: list, negative_prompts: list, batch_size: int = 1):
        # In unconditional scaling is not $1$ get the embeddings for empty prompts (no conditioning).
        if uncond_scale != 1. and len(negative_prompts) == 0:
            un_cond = self.get_empty_embedding()
        elif len(negative_prompts) != 0:
            un_cond = self.model.get_text_conditioning(negative_prompts)
        else:
            un_cond = None

        # Get the prompt embeddings
        cond = self.model.get_text_conditioning(prompts)

        return un_cond, cond

    def get_image_from_latent(self, x: torch.Tensor):
        return self.model.autoencoder_decode(x)

    @torch.no_grad()
    def generate_images_latent_from_embeddings(self, *,
                                               seed: int = 0,
                                               batch_size: int = 1,
                                               embedded_prompt: torch.Tensor,
                                               null_prompt: torch.Tensor,
                                               h: int = 512, w: int = 512,
                                               uncond_scale: float = 7.5,
                                               low_vram: bool = False,
                                               noise_fn=torch.randn,
                                               temperature: float = 1.0,
                                               ):
        """
        :param seed: the seed to use when generating the images
        :param dest_path: is the path to store the generated images
        :param batch_size: is the number of images to generate in a batch
        :param prompt: is the prompt to generate images with
        :param h: is the height of the image
        :param w: is the width of the image
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param low_vram: whether to limit VRAM usage
        """

        # check null_prompt, raise exception if None
        if null_prompt is None:
            raise Exception("Null prompt cannot be None.")

        # Number of channels in the image
        c = 4
        # Image to latent space resolution reduction
        f = 8

        if seed == 0:
            seed = time.time_ns() % 2 ** 32

        set_seed(seed)
        # Adjust batch size based on VRAM availability
        if low_vram:
            batch_size = 1

        # AMP auto casting
        autocast = get_autocast()
        with autocast:

            # [Sample in the latent space](../sampler/index.html).
            # `x` will be of shape `[batch_size, c, h / f, w / f]`
            x = self.sampler.sample(cond=embedded_prompt,
                                    shape=[batch_size, c, h // f, w // f],
                                    uncond_scale=uncond_scale,
                                    uncond_cond=null_prompt,
                                    noise_fn=noise_fn,
                                    temperature=temperature)

            return x

    def paint(self,
              orig: torch.Tensor,
              cond: torch.Tensor,
              t_index: int,
              uncond_scale: float = 1.0,
              un_cond: Optional[torch.Tensor] = None,
              mask: Optional[torch.Tensor] = None,
              orig_noise: Optional[torch.Tensor] = None):

        orig_2 = None
        # If we have a mask and noise, it's in-painting
        if mask is not None and orig_noise is not None:
            orig_2 = orig
        # Add noise to the original image
        x = self.sampler.q_sample(orig, t_index, noise=orig_noise)
        # Reconstruct from the noisy image
        x = self.sampler.paint(x, cond, t_index,
                               orig=orig_2,
                               mask=mask,
                               orig_noise=orig_noise,
                               uncond_scale=uncond_scale,
                               uncond_cond=un_cond)

        return x

    def load_model(self, model_path=LATENT_DIFFUSION_PATH, batch_size: int = 1):
        with section(f'Latent Diffusion model loading, from {model_path}'):
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()

            # set empty embedding
            self.cache_empty_embedding(batch_size)

    def unload_model(self):
        del self.model
        torch.cuda.empty_cache()

    def save_model(self, model_path=LATENT_DIFFUSION_PATH):
        with section(f'Latent Diffusion model saving, to {model_path}'):
            torch.save(self.model, model_path)

    def initialize_from_model(self, model: LatentDiffusion):

        self.model = model
        self.initialize_sampler()

    def initialize_from_saved(self, model_path=LATENT_DIFFUSION_PATH):

        self.load_model(model_path)
        self.initialize_sampler()

    def initialize_script(self, autoencoder=None, clip_text_embedder=None, unet_model=None, force_submodels_init=False,
                          path=None):
        """You can initialize the autoencoder, CLIP and UNet models externally and pass them to the script.
        Use the methods:
            stable_diffusion.utils.model.initialize_autoencoder,
            stable_diffusion.utils.model.initialize_clip_embedder
            and
            stable_diffusion.utils.model.initialize_unet
        to initialize them.
        If you don't initialize them externally, the script will initialize them internally.
        Args:
            autoencoder (Autoencoder, optional): the externally initialized autoencoder. Defaults to None.
            clip_text_embedder (CLIPTextEmbedder, optional): the externally initialized autoencoder. Defaults to None.
            unet_model (UNetModel, optional): the externally initialized autoencoder. Defaults to None.
        """
        self.initialize_latent_diffusion(autoencoder, clip_text_embedder, unet_model,
                                         force_submodels_init=force_submodels_init, path=path)
        self.initialize_sampler()

    def initialize_latent_diffusion(self, autoencoder, clip_text_embedder, unet_model, force_submodels_init=False,
                                    path=None, batch_size=1):
        try:
            self.model = initialize_latent_diffusion(
                path=path,
                device=self.device_id,
                autoencoder=autoencoder,
                clip_text_embedder=clip_text_embedder,
                unet_model=unet_model,
                force_submodels_init=force_submodels_init,
            )
            self.initialize_sampler()
            # Move the model to device
            # self.model.to(self.device)

            # set empty embedding
            self.cache_empty_embedding(batch_size)

        except EOFError:
            raise ModelLoadError(
                "Stable Diffusion model couldn't be loaded. Check that the .ckpt file exists in the specified location (path), and that it is not corrupted.")

    def initialize_sampler(self):
        if self.sampler_name == 'ddim':
            self.sampler = DDIMSampler(self.model,
                                       n_steps=self.n_steps,
                                       ddim_eta=self.ddim_eta)
        elif self.sampler_name == 'ddpm':
            self.sampler = DDPMSampler(self.model)

    def cache_empty_embedding(self, batch_size: int = 1):
        self.empty_embedding = self.model.get_text_conditioning(batch_size * [""])

    def get_empty_embedding(self):
        return self.empty_embedding

