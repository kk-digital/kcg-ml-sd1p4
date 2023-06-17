#%%
import os
from typing import Callable

import torch
import time
from tqdm import tqdm




noise_seeds = [
    2982,
    4801,
    1995,
    3598,
    987,
    3688,
    8872,
    762
]
#
# CURRENTLY adding kwargs for custom noise function for the samplers
# TODO adapt the script so that it creates a folder for the outputs of each kind of noise
# TODO adapt the script so that it generates images for each kind of noise
# TODO add kwargs for some samplers' parameters

def log_normal(shape, device, mu=0.0, sigma=0.25):
    return torch.exp(mu + sigma*torch.randn(shape, device=device))

def normal(shape, device='cuda:0', mu=0.0, sigma=1.0):
    return torch.normal(mu, sigma, size=shape, device=device)

DISTRIBUTIONS = {
    'Normal': dict(loc=0.0, scale=1.0),
    'Cauchy': dict(loc=0.0, scale=1.0), 
    'Gumbel': dict(loc=1.0, scale=2.0), 
    'Laplace': dict(loc=0.0, scale=1.0), 
    'Uniform': dict(low=0.0, high=1.0)
}

OUTPUT_DIR = os.path.abspath('./output/noise-tests/')

CLEAR_OUTPUT_DIR = True

def get_all_torch_distributions() -> tuple[list[str], list[type]]:
    
    torch_distributions_names = torch.distributions.__all__
    
    torch_distributions = [
        torch.distributions.__dict__[torch_distribution_name] \
            for torch_distribution_name in torch_distributions_names
            ]
    
    return torch_distributions_names, torch_distributions

def get_torch_distribution_from_name(name: str) -> type:
    return torch.distributions.__dict__[name]

def build_noise_samplers(distributions: dict[str, dict[str, float]]) -> dict[str, Callable]:
    noise_samplers = {
        k: lambda shape, device = None: get_torch_distribution_from_name(k)(**v).sample(shape).to(device) \
                       for k, v in distributions.items()
                       }
    return noise_samplers

def create_folder_structure(distributions_dict: dict[str, dict[str, float]], root_dir: str = OUTPUT_DIR) -> None:
    for distribution_name in distributions_dict.keys():
        
        distribution_outputs = os.path.join(root_dir, distribution_name)
        try:
            os.makedirs(distribution_outputs, exist_ok=True)
        except Exception as e:
            print(e)

NOISE_SAMPLERS = build_noise_samplers(DISTRIBUTIONS)

# %%
a, b = get_all_torch_distributions()
# %%
normal = get_torch_distribution_from_name('Normal')
# %%
noise_samplers = build_noise_samplers(DISTRIBUTIONS)
# %%

# %%
create_folder_structure(DISTRIBUTIONS)

#%%


# %%
