import time

import random
import numpy as np

import torch
import torchvision
from torch.mps import current_allocated_memory as mps_current_allocated_memory

import PIL
from PIL import Image
from torchvision.transforms import ToPILImage

from pathlib import Path
from typing import Union, BinaryIO, List, Optional
import hashlib

from stable_diffusion.utils.logger import logger


class SectionManager:
    def __init__(self, name: str):
        self.t0 = None
        self.t1 = None
        self.section_name = name

    def __enter__(self):
        print(f"Starting section: {self.section_name}...")
        self.t0 = time.time()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.t1 = time.time()
        print(f"Finished section: {self.section_name} in {(self.t1 - self.t0):.2f} seconds\n")


def calculate_sha256(tensor):
    if tensor.device == "cpu":
        tensor_bytes = tensor.numpy().tobytes()  # Convert tensor to a byte array
    else:
        tensor_bytes = tensor.cpu().numpy().tobytes()  # Convert tensor to a byte array
    sha256_hash = hashlib.sha256(tensor_bytes)
    return sha256_hash.hexdigest()


def to_pil(image):
    return ToPILImage()(torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0))


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


def show_image_grid(
        tensor: Union[torch.Tensor, List[torch.Tensor]],
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
    return im


def load_img(path: str):
    """
    ### Load an image

    This loads an image from a file and returns a PyTorch tensor.

    :param path: is the path of the image
    """
    # Open Image
    image = Image.open(path).convert("RGB")
    # Get image size
    w, h = image.size
    # Resize to a multiple of 32
    w = w - w % 32
    h = h - h % 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    # Convert to numpy and map to `[-1, 1]` for `[0, 255]`
    image = np.array(image).astype(np.float32) * (2. / 255.0) - 1
    # Transpose to shape `[batch_size, channels, height, width]`
    image = image[None].transpose(0, 3, 1, 2)
    # Convert to torch
    return torch.from_numpy(image)


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


def get_device(device=None):
    device_priority = {
        'cuda': [torch.cuda.is_available,
                 lambda x: logger.info(f'Using CUDA device {x.index}: {torch.cuda.get_device_name(x)}')],
        'mps': [torch.backends.mps.is_available, lambda _: logger.info('Using MPS device')],
        'cpu': [lambda: True, lambda x: logger.warning(
            f'You are running this script without CUDA or MPS (current device: {x}). It may be very slow')
]
    }

    if device is None or device not in device_priority:
        for _device, (availability_check, log_msg) in device_priority.items():
            if availability_check():
                log_msg(_device)
                return torch.device(_device)

    availability_check, log_msg = device_priority[device]
    if availability_check():
        log_msg(device)
        return torch.device(device)

    raise Exception(f'Device {device if device else "any"} not available.')


def get_memory_status(device=None):
    logger.info(f'Using device: {device}')
    if device == None:
        logger.warning('No device specified.')
    elif device.type == 'cuda':
        t = torch.cuda.get_device_properties(device).total_memory // 1024 ** 2
        r = torch.cuda.memory_reserved(device) // 1024 ** 2
        a = torch.cuda.memory_allocated(device) // 1024 ** 2
        f = t - (r + a)
        logger.info(f'Total: {t} MiB\nFree: {f} MiB\nReserved: {r} MiB\nAllocated: {a} MiB')
    elif device.type == 'mps':
        # get total memory from cpu
        a = mps_current_allocated_memory() // 1024 ** 2
        logger.info(f'Total: ? MiB - Free: ? MiB - Reserved: ? MiB - Allocated: {a} MiB')
    else:
        logger.warning(f'Unknown device: {device}')


def get_autocast(force_cpu: bool = False):
    """
    ### Get autocast
    """
    if torch.cuda.is_available() and not force_cpu:
        return torch.autocast(device_type='cuda')

    if torch.backends.mps.is_available() and not force_cpu:
        logger.warning("MPS device does not support autocast.")
        return torch.autocast(device_type='cpu')

    logger.warning("You are running this script without CUDA. It may be very slow.")
    return torch.autocast(device_type='cpu')


def set_seed(seed: int):
    """
    ### Set random seeds
    """
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
