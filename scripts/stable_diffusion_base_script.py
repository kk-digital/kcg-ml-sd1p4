
import torch

from stable_diffusion.sampler.ddim import DDIMSampler
from stable_diffusion.sampler.ddpm import DDPMSampler
from stable_diffusion.util import load_model, get_device

from typing import Union

from pathlib import Path

class StableDiffusionBaseScript:
    def __init__(self, *, checkpoint_path: Union[str, Path],
                 ddim_steps: int = 50,
                 ddim_eta: float = 0.0,
                 force_cpu: bool = False,
                 sampler_name: str='ddim',
                 n_steps: int = 50,
                 cuda_device: str = 'cuda:0',
                 ):
        """
        :param checkpoint_path: is the path of the checkpoint
        :param sampler_name: is the name of the [sampler](../sampler/index.html)
        :param n_steps: is the number of sampling steps
        :param ddim_eta: is the [DDIM sampling](../sampler/ddim.html) $\eta$ constant
        """
        self.ddim_steps = ddim_steps

        device_id = get_device(force_cpu, cuda_device)

        # Load [latent diffusion model](../latent_diffusion.html)
        self.model = load_model(checkpoint_path, device_id)
        # Get device or force CPU if requested
        self.device = torch.device(device_id)

        # Move the model to device
        self.model.to(self.device)

        # Initialize [sampler](../sampler/index.html)
        if sampler_name == 'ddim':
            self.sampler = DDIMSampler(self.model,
                                       n_steps=n_steps,
                                       ddim_eta=ddim_eta)
        elif sampler_name == 'ddpm':
            self.sampler = DDPMSampler(self.model)