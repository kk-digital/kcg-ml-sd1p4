import os, sys
sys.path.append(os.path.abspath(''))

import torch
import time

from stable_diffusion2.sampler.ddim import DDIMSampler
from stable_diffusion2.sampler.ddpm import DDPMSampler
from stable_diffusion2.utils.model import initialize_latent_diffusion
from stable_diffusion2.utils.utils import check_device, get_device, load_img, get_memory_status, set_seed, get_autocast
from stable_diffusion2.latent_diffusion import LatentDiffusion
from stable_diffusion2.sampler import DiffusionSampler
from stable_diffusion2.constants import LATENT_DIFFUSION_PATH
from stable_diffusion2.utils.utils import SectionManager as section
from typing import Union, Optional
from pathlib import Path

class ModelLoadError(Exception):
    pass

class StableDiffusion:
    model: LatentDiffusion
    sampler: DiffusionSampler

    def __init__(self, *,
                 ddim_steps: int = 50,
                 ddim_eta: float = 0.0,
                 force_cpu: bool = False,
                 sampler_name: str='ddim',
                 n_steps: int = 50,
                 device = None,
                 model: LatentDiffusion = None,
                 ):
        """
        :param checkpoint_path: is the path of the checkpoint
        :param sampler_name: is the name of the [sampler](../sampler/index.html)
        :param n_steps: is the number of sampling steps
        :param ddim_eta: is the [DDIM sampling](../sampler/ddim.html) $\eta$ constant
        """
        self.device = check_device(device)
        self.model = model
        self.ddim_steps = ddim_steps
        self._ddim_eta = ddim_eta
        self.force_cpu = force_cpu
        self.sampler_name = sampler_name
        self._sampler = None
        self.n_steps = n_steps
        if self.model is None:
            
            print("WARNING: LatentDiffusion model not given.")
            
            self.model = LatentDiffusion(linear_start=0.00085,
            linear_end=0.0120,
            n_steps=1000,
            latent_scaling_factor=0.18215,
            device = self.device)

    @property
    def sampler(self):
        self.initialize_sampler()
        return self._sampler

    @property
    def ddim_eta(self):
        return self._ddim_eta
    @ddim_eta.setter
    def ddim_eta(self, value):
        self._ddim_eta = value
        if self.sampler_name == 'ddim':
            self.initialize_sampler()

    def encode_image(self, orig_img: str, batch_size: int = 1):
        """
        Encode an image in the latent space
        """
        orig_image = load_img(orig_img).to(self.device)
        # Encode the image in the latent space and make `batch_size` copies of it
        orig = self.model.autoencoder_encode(orig_image).repeat(batch_size, 1, 1, 1)
        # with section("Encoding image"):
        #     print(get_memory_status())
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

    def get_text_conditioning(self, uncond_scale: float, prompts: list, batch_size: int = 1):
        # In unconditional scaling is not $1$ get the embeddings for empty prompts (no conditioning).
        if uncond_scale != 1.0:
            un_cond = self.model.get_text_conditioning(batch_size * [""])
        else:
            un_cond = None

        # Get the prompt embeddings
        cond = self.model.get_text_conditioning(prompts)

        return un_cond, cond

    def decode_image(self, x: torch.Tensor):
        # with section("decoding image"):
        #     print(get_memory_status())
        return self.model.autoencoder_decode(x)

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

    def load_model(self, model_path = LATENT_DIFFUSION_PATH):
        with section(f'Latent Diffusion model loading, from {model_path}'):
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()

    def unload_model(self):
        # del self.model.autoencoder.encoder
        # del self.model.autoencoder.decoder
        self.model.first_stage_model.unload_submodels()
        self.model.cond_stage_model.unload_submodels()
        del self.model.model
        torch.cuda.empty_cache()

    def save_model(self, model_path = LATENT_DIFFUSION_PATH):
        with section(f'Latent Diffusion model saving, to {model_path}'):
            torch.save(self.model, model_path)
    
    def initialize_from_model(self, model: LatentDiffusion):

        self.model = model
        self.initialize_sampler()  

    def initialize_script(self, autoencoder = None, clip_text_embedder = None, unet_model = None, force_submodels_init = False, path = None):
        """You can initialize the autoencoder, CLIP and UNet models externally and pass them to the script.
        Use the methods: 
            stable_diffusion.utils.model.initialize_autoencoder,
            stable_diffusion.utils.model.initialize_clip_embedder and 
            stable_diffusion.utils.model.initialize_unet to initialize them.
        If you don't initialize them externally, the script will initialize them internally.
        Args:
            autoencoder (Autoencoder, optional): the externally initialized autoencoder. Defaults to None.
            clip_text_embedder (CLIPTextEmbedder, optional): the externally initialized autoencoder. Defaults to None.
            unet_model (UNetModel, optional): the externally initialized autoencoder. Defaults to None.
        """
        raise DeprecationWarning("This method is deprecated. Use initialize_latent_diffusion instead.")
        self.initialize_latent_diffusion(autoencoder, clip_text_embedder, unet_model, force_submodels_init=force_submodels_init, path=path)
        self.initialize_sampler()

    def initialize_latent_diffusion(self, path = None, autoencoder = None, clip_text_embedder = None, unet_model = None, force_submodels_init = False):
        try:
            self.model = initialize_latent_diffusion(
                path=path,
                device=self.device,
                autoencoder=autoencoder,
                clip_text_embedder=clip_text_embedder,
                unet_model = unet_model,
                force_submodels_init=force_submodels_init
            )

            self.initialize_sampler()
        except EOFError:
                raise ModelLoadError("Stable Diffusion model couldn't be loaded. Check that the .ckpt file exists in the specified location (path), and that it is not corrupted.")

    def initialize_sampler(self):
        if self.sampler_name == 'ddim':
            self._sampler = DDIMSampler(self.model,
                                       n_steps=self.n_steps,
                                       ddim_eta=self._ddim_eta)
        elif self.sampler_name == 'ddpm':
            self._sampler = DDPMSampler(self.model)

    @torch.no_grad()
    def generate_images(self, *,
                seed: int = 0,
                batch_size: int = 1,
                prompt: str,
                h: int = 512, w: int = 512,
                uncond_scale: float = 7.5,
                low_vram: bool = False,
                noise_fn = torch.randn,
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
            seed = time.time_ns() % 2**32
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
            un_cond, cond = self.get_text_conditioning(uncond_scale, prompts, batch_size)
            # [Sample in the latent space](../sampler/index.html).
            # `x` will be of shape `[batch_size, c, h / f, w / f]`
            # with section("sampling"):
            x = self.sampler.sample(cond=cond,
                                        shape=[batch_size, c, h // f, w // f],
                                        uncond_scale=uncond_scale,
                                        uncond_cond=un_cond,
                                        noise_fn=noise_fn,
                                        temperature=temperature)
            return self.decode_image(x)

    @torch.no_grad()
    def decode(self, x: torch.Tensor):
        
        # AMP auto casting
        autocast = get_autocast()
        
        with autocast:
            return self.decode_image(x)        
    
    @torch.no_grad()
    def encode(self, image: torch.Tensor):
        
        # AMP auto casting
        autocast = get_autocast()

        with autocast:
            return self.model.autoencoder_encode(image.to(self.device))            