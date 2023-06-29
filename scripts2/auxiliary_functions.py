from pathlib import Path
from typing import Union, BinaryIO, List, Optional
from typing import Callable

import PIL
import numpy as np
import torch
import torchvision

from PIL import Image

def save_images(images: torch.Tensor, dest_path: str, img_format: str = 'jpeg'):
    """
    ### Save a images

    :param images: is the tensor with images of shape `[batch_size, channels, height, width]`
    :param dest_path: is the folder to save images in
    :param img_format: is the image format
    """

    # Map images to `[0, 1]` space and clip
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    # Transpose to `[batch_size, height, width, channels]` and convert to numpy
    images = images.cpu()
    images = images.permute(0, 2, 3, 1)
    images = images.float().numpy()

    # Save images
    for i, img in enumerate(images):
        img = Image.fromarray((255. * img).astype(np.uint8))
        img.save(dest_path, format=img_format)

def save_image_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[str, Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = torchvision.utils.make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format) 

def get_torch_distribution_from_name(name: str) -> type:
    if name == 'Logistic':
        def logistic_distribution(loc, scale):
            base_distribution = torch.distributions.Uniform(0, 1)
            transforms = [torch.distributions.transforms.SigmoidTransform().inv, torch.distributions.transforms.AffineTransform(loc=loc, scale=scale)]
            logistic = torch.distributions.TransformedDistribution(base_distribution, transforms)
            return logistic
        return logistic_distribution
    return torch.distributions.__dict__[name]

def get_all_torch_distributions() -> tuple[list[str], list[type]]:
    
    torch_distributions_names = torch.distributions.__all__
    
    torch_distributions = [
        torch.distributions.__dict__[torch_distribution_name] \
            for torch_distribution_name in torch_distributions_names
            ]
    
    return torch_distributions_names, torch_distributions

def build_noise_samplers(distributions: dict[str, dict[str, float]]) -> dict[str, Callable]:
    noise_samplers = {
        k: lambda shape, device = None: get_torch_distribution_from_name(k)(**v).sample(shape).to(device) \
                       for k, v in distributions.items()
                       }
    return noise_samplers