import os
import sys
import time
from typing import Optional

import torch

from utility.utils_logger import logger

sys.path.append(os.path.abspath(""))
from stable_diffusion.utils_backend import get_device, get_autocast, set_seed
from stable_diffusion.utils_image import load_img
from stable_diffusion.sampler.ddim import DDIMSampler
from stable_diffusion.sampler.ddpm import DDPMSampler
from stable_diffusion.utils_model import initialize_latent_diffusion
from stable_diffusion.latent_diffusion import LatentDiffusion
from stable_diffusion.sampler.diffusion import DiffusionSampler
from stable_diffusion.model_paths import LATENT_DIFFUSION_PATH
from utility.labml.monit import section


class ModelLoadError(Exception):
    pass


class StableDiffusion:
    model: LatentDiffusion
    sampler: DiffusionSampler

    def __init__(
            self,
            *,
            device=None,
            model: LatentDiffusion = None,
            ddim_steps: int = 50,
            ddim_eta: float = 0.0,
            force_cpu: bool = False,
            sampler_name: str = "ddim",
            n_steps: int = 50,
    ):
        """
        :param checkpoint_path: is the path of the checkpoint
        :param sampler_name: is the name of the [sampler](../sampler/index.html)
        :param n_steps: is the number of sampling steps
        :param ddim_eta: is the [DDIM sampling](../sampler/ddim.html) $\eta$ constant
        """

        self._device = get_device(device)
        self._model = model
        self._ddim_steps = ddim_steps
        self._ddim_eta = ddim_eta
        self._sampler_name = sampler_name
        self.sampler = None
        self._n_steps = n_steps

        if self._model is None:
            logger.warning("`LatentDiffusion` model is `None` given. Initialize one with the appropriate method.")
        elif type(self._model) == LatentDiffusion:
            logger.info("LatentDiffusion model given. Initializing sampler.")
            self.model = self._model

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = get_device(value)
        self._model.to(self._device)
        self.initialize_sampler()

    @property
    def ddim_eta(self):
        return self._ddim_eta

    @ddim_eta.setter
    def ddim_eta(self, value):
        self._ddim_eta = value
        if self.sampler_name == "ddim":
            self.initialize_sampler()

    @property
    def ddim_steps(self):
        return self._ddim_steps

    @ddim_steps.setter
    def ddim_steps(self, value):
        self._ddim_steps = value
        if self.sampler_name == "ddim":
            self.initialize_sampler()

    @property
    def n_steps(self):
        return self._n_steps

    @n_steps.setter
    def n_steps(self, value):
        self._n_steps = value
        if self.sampler_name == "ddpm":
            self.initialize_sampler()

    @property
    def sampler_name(self):
        return self._sampler_name

    @sampler_name.setter
    def sampler_name(self, value):
        self._sampler_name = value
        self.initialize_sampler()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
        self._model.to(self._device)
        self.initialize_sampler()

    def encode_image(self, orig_img: str, batch_size: int = 1):
        """
        Encode an image in the latent space
        """
        orig_image = load_img(orig_img).to(self.device)
        # Encode the image in the latent space and make `batch_size` copies of it
        orig = self.model.autoencoder_encode(orig_image).repeat(batch_size, 1, 1, 1)
        return orig

    @torch.no_grad()
    def encode(self, image: torch.Tensor, batch_size: int = 1):
        # AMP auto casting
        autocast = get_autocast()

        with autocast:
            return self.model.autoencoder_encode(image.to(self.device))

    def get_image_from_latent(self, x: torch.Tensor):
        return self.model.autoencoder_decode(x)

    @torch.no_grad()
    def decode(self, x: torch.Tensor):
        # AMP auto casting
        autocast = get_autocast()

        with autocast:
            return self.get_image_from_latent(x.to(self.device))

    def prepare_mask(self, mask: Optional[torch.Tensor], orig: torch.Tensor):
        # If `mask` is not provided,
        # we set a sample mask to preserve the bottom half of the image
        if mask is None:
            mask = torch.zeros_like(orig, device=self.device)
            mask[:, :, mask.shape[2] // 2:, :] = 1.0
        else:
            mask = mask.to(self.device)

        return mask

    def calc_strength_time_step(self, strength: float):
        # Get the number of steps to diffuse the original
        t_index = int(strength * self.ddim_steps)

        return t_index

    def get_text_conditioning(
            self, uncond_scale: float, prompts: list, negative_prompts: list, batch_size: int = 1
    ):
        # In unconditional scaling is not $1$ get the embeddings for empty prompts (no conditioning).
        if uncond_scale != 1.0 and len(negative_prompts) == 0:
            un_cond = self.model.get_text_conditioning(batch_size * [""])
        elif len(negative_prompts) != 0:
            un_cond = self.model.get_text_conditioning(negative_prompts)
        else:
            un_cond = None

        # Get the prompt embeddings
        cond = self.model.get_text_conditioning(prompts)

        return un_cond, cond

    def paint(
            self,
            orig: torch.Tensor,
            cond: torch.Tensor,
            t_index: int,
            uncond_scale: float = 1.0,
            un_cond: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            orig_noise: Optional[torch.Tensor] = None,
    ):
        orig_2 = None
        # If we have a mask and noise, it's in-painting
        if mask is not None and orig_noise is not None:
            orig_2 = orig
        # Add noise to the original image
        x = self.sampler.q_sample(orig, t_index, noise=orig_noise)
        # Reconstruct from the noisy image
        x = self.sampler.paint(
            x,
            cond,
            t_index,
            orig=orig_2,
            mask=mask,
            orig_noise=orig_noise,
            uncond_scale=uncond_scale,
            uncond_cond=un_cond,
        )

        return x

    def load_model(self, latent_diffusion_path=LATENT_DIFFUSION_PATH):
        with section(f"Latent Diffusion model loading, from {latent_diffusion_path}"):
            self.model = self.quick_initialize().load(latent_diffusion_path=latent_diffusion_path)
            return self.model

    def save_model(self, latent_diffusion_path=LATENT_DIFFUSION_PATH):
        with section(f"Latent Diffusion model saving, to {latent_diffusion_path}"):
            self.model.save(latent_diffusion_path=latent_diffusion_path)

    def unload_model(self):
        self.model.first_stage_model.unload_submodels()
        self.model.cond_stage_model.unload_submodels()
        self.model.unload_unet()
        torch.cuda.empty_cache()

    def quick_initialize(self):
        self.model = LatentDiffusion(
            device=self.device,
        )
        self.initialize_sampler()
        return self.model

    def initialize_latent_diffusion(
            self,
            path=None,
            autoencoder=None,
            clip_text_embedder=None,
            unet_model=None,
            force_submodels_init=False,
    ):
        try:
            self.model = initialize_latent_diffusion(
                path=path,
                device=self.device,
                autoencoder=autoencoder,
                clip_text_embedder=clip_text_embedder,
                unet_model=unet_model,
                force_submodels_init=force_submodels_init,
            )

            self.initialize_sampler()
            return self.model
        except EOFError:
            raise ModelLoadError(
                "Stable Diffusion model couldn't be loaded. Check that the .ckpt file exists in the specified location (path), and that it is not corrupted."
            )

    def initialize_sampler(self):
        if self.sampler_name == "ddim":
            self.sampler = DDIMSampler(
                self.model, n_steps=self.n_steps, ddim_eta=self.ddim_eta
            )
        elif self.sampler_name == "ddpm":
            self.sampler = DDPMSampler(self.model)

    @torch.no_grad()
    def generate_images(
            self,
            *,
            seed: int = 0,
            batch_size: int = 1,
            prompt: str,
            negative_prompt: str,
            h: int = 512,
            w: int = 512,
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
        # Make a batch of prompts
        prompts = batch_size * [prompt]
        # AMP auto casting
        autocast = get_autocast()
        with autocast:
            # with section("getting text cond"):
            un_cond, cond = self.get_text_conditioning(
                uncond_scale, prompts, negative_prompt, batch_size
            )
            # [Sample in the latent space](../sampler/index.html).
            # `x` will be of shape `[batch_size, c, h / f, w / f]`
            # with section("sampling"):
            x = self.sampler.sample(
                cond=cond,
                shape=[batch_size, c, h // f, w // f],
                uncond_scale=uncond_scale,
                uncond_cond=un_cond,
                noise_fn=noise_fn,
                temperature=temperature,
            )
            return self.get_image_from_latent(x)

    @torch.no_grad()
    def generate_images_latent_from_embeddings(
            self,
            *,
            seed: int = 0,
            batch_size: int = 1,
            embedded_prompt: torch.Tensor,
            null_prompt: torch.Tensor,
            h: int = 512,
            w: int = 512,
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

        # Make a batch of prompts
        prompts = batch_size * [embedded_prompt]
        cond = torch.cat(prompts, dim=0)
        null_prompts = batch_size * [null_prompt]
        uncond_cond = torch.cat(null_prompts, dim=0)

        # AMP auto casting
        autocast = get_autocast()
        with autocast:
            # [Sample in the latent space](../sampler/index.html).
            # `x` will be of shape `[batch_size, c, h / f, w / f]`
            x = self.sampler.sample(
                cond=cond,
                shape=[batch_size, c, h // f, w // f],
                uncond_scale=uncond_scale,
                uncond_cond=uncond_cond,
                noise_fn=noise_fn,
                temperature=temperature,
            )

            return x
