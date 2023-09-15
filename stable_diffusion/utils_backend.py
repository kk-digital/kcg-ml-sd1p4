import contextlib
import sys

import numpy as np
import torch
from torch.mps import current_allocated_memory as mps_current_allocated_memory

from utility.utils_logger import logger



def get_device(device=None):
    device_priority = {
        'cuda': [torch.cuda.is_available,
                 lambda x: logger.debug(f'Using CUDA device {torch.cuda.get_device_name(x)}')],
        'mps': [torch.backends.mps.is_available, lambda _: logger.debug('Using MPS device')],
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
    if device is None:
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

def without_autocast(disable=False):
    return torch.autocast("cuda", enabled=False) if torch.is_autocast_enabled() and not disable else contextlib.nullcontext()

def set_seed(seed):
    """
    ### Set random seeds
    """
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cuda_device_string():
    # if shared.cmd_opts.device_id is not None:
    #     return f"cuda:{shared.cmd_opts.device_id}"

    return "cuda"


def has_mps() -> bool:
    if sys.platform != "darwin":
        return False
    else:
        return torch.backends.mps.is_available() and torch.backends.mps.is_built()


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(get_cuda_device_string()):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    if has_mps():
        from torch.mps import empty_cache
        empty_cache()
