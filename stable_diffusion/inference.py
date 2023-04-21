#! python
# copyright (c) 2022 C.Y. Wong, myByways.com simplified Stable Diffusion v0.1

import argparse
import os, sys, time
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from contextlib import nullcontext
from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.ddim import DDIMSampler
from transformers import logging

def parse_args():
    parser = argparse.ArgumentParser(description='Simplified Stable Diffusion')
    parser.add_argument('--prompt', type=str, nargs='+', required=True, help='one or more prompts')
    parser.add_argument('--negative', type=str, nargs='+', default=None, help='one or more negative prompts')
    parser.add_argument('--H', type=int, default=512, help='image height')
    parser.add_argument('--W', type=int, default=512, help='image width')
    parser.add_argument('--f', type=int, default=8, help='downsampling factor')
    parser.add_argument('--fixed_code', type=int, default=0, help='1 for repeatable results')
    parser.add_argument('--seed', type=int, default=None, help='random seed for stable diffusion')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='noise level for DDIM sampler')
    parser.add_argument('--plms', type=int, default=0, help='1 for PLMS sampler')
    parser.add_argument('--n_iter', type=int, default=1, help='number of iterations')
    parser.add_argument('--scale', type=float, default=7.5, help='guidance scale for unconditional samples')
    parser.add_argument('--ddim_steps', type=int, default=50, help='number of DDIM steps')
    parser.add_argument('--init_img', type=str, default=None, help='path to the initial image for image-to-image generation')
    parser.add_argument('--strength', type=float, default=0.75, help='strength for conditioning on initial image for image-to-image generation')
    parser.add_argument('--outdir', type=str, default='/output/', help='output directory')
    parser.add_argument('--history', type=str, default='history.txt', help='history file')
    parser.add_argument('--config', type=str, default='configs/stable-diffusion/v1-inference.yaml', help='config file')
    parser.add_argument('--ckpt', type=str, default='/input/model/sd-v1-4.ckpt', help='checkpoint file')
    parser.add_argument('--low-vram', action='store_true', default=False, help='reduce GPU memory usage')

    return parser.parse_args()

args = parse_args()

HEIGHT = args.H
WIDTH = args.W
FACTOR = args.f
FIXED = args.fixed_code
NOISE = args.ddim_eta
PLMS = args.plms
ITERATIONS = args.n_iter
SCALE = args.scale
STEPS = args.ddim_steps
IMAGE = args.init_img
STRENGTH = args.strength
FOLDER = args.outdir
HISTORY = args.history
CONFIG = args.config
CHECKPOINT = args.ckpt
LOW_VRAM = args.low_vram

PROMPTS = args.prompt
NEGATIVES = args.negative

def seed_pre():
    if not FIXED:
        if args.seed is None:
            seed = int(time.time() * 1000) % (2**32)
            print(f"Setting random stable diffusion seed: {seed}")
        else:
            seed=args.seed
    else:
        seed = FIXED_SEED
    return seed

def seed_post(device):
    if FIXED:
        seed_everything(SEED)
        return torch.randn([1, 4, HEIGHT // FACTOR, WIDTH // FACTOR], device='cpu').to(torch.device(device.type))
    return None

def load_model(config, ckpt=CHECKPOINT):
    pl_sd = torch.load(ckpt, map_location='cpu')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    return model

def set_device(model, low_vram=False):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        precision = torch.autocast
        if low_vram:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        precision = torch.autocast
    model.to(device.type)
    model.eval()
    return device, precision

def load_image(image_file):
    image = Image.open(image_file).convert('RGB')
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.0

def setup_sampler(model):
    global NOISE
    if IMAGE:
        image = load_image(IMAGE).to(model.device.type)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(image))
        sampler = DDIMSampler(model)
        sampler.make_schedule(ddim_num_steps=STEPS, ddim_eta=NOISE, verbose=False)
        t_enc = int(STRENGTH * STEPS)
        sampler.t_enc = t_enc
        sampler.z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(model.device.type))
    elif PLMS:
        sampler = PLMSSampler(model)
        NOISE = 0
    else:
        sampler = DDIMSampler(model)
    return sampler

def get_prompts():
    global NEGATIVES
    if NEGATIVES is None:
        NEGATIVES = [''] * len(PROMPTS)
    else:
        NEGATIVES.extend([''] * (len(PROMPTS)-len(NEGATIVES)))
    return zip(PROMPTS, NEGATIVES)

def generate_samples(model, sampler, prompt, negative, start):
    uncond = model.get_learned_conditioning(negative) if SCALE != 1.0 else None
    cond = model.get_learned_conditioning(prompt)
    if IMAGE:
        samples = sampler.decode(sampler.z_enc, cond, sampler.t_enc,
            unconditional_guidance_scale=SCALE, unconditional_conditioning=uncond)
    else:
        shape = [4, HEIGHT // FACTOR, WIDTH // FACTOR]
        samples, _ = sampler.sample(S=STEPS, conditioning=cond, batch_size=1,
            shape=shape, verbose=False, unconditional_guidance_scale=SCALE,
            unconditional_conditioning=uncond, eta=NOISE, x_T=start)
    x_samples = model.decode_first_stage(samples)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    return x_samples

def save_image(image):
    name = f'{time.strftime("%Y%m%d_%H%M%S")}.png'
    image = 255. * rearrange(image.cpu().numpy(), 'c h w -> h w c')
    img = Image.fromarray(image.astype(np.uint8))
    img.save(os.path.join(FOLDER, name))
    return name

def save_history(name, prompt, negative):
    with open(os.path.join(FOLDER, HISTORY), 'a') as history:
        history.write(f'{name} -> {"PLMS" if PLMS else "DDIM"}, Scale={SCALE}, Steps={STEPS}, Noise={NOISE}')
        if IMAGE:
            history.write(f', Image={IMAGE}, Strength={STRENGTH}')
        if len(negative):
            history.write(f'\n + {prompt}\n - {negative}\n')
        else:
            history.write(f'\n + {prompt}\n')

def main():
    tic1 = time.time()
    logging.set_verbosity_error()
    os.makedirs(FOLDER, exist_ok=True)

    seed_pre()
    config = OmegaConf.load(CONFIG)
    model = load_model(config)
    device, precision_scope = set_device(model)
    sampler = setup_sampler(model)
    start_code = seed_post(device)

    toc1 = time.time()
    print(f'*** Model setup time: {(toc1 - tic1):.2f}s')

    counter = 0
    with torch.no_grad():
        with precision_scope(device.type):
            with model.ema_scope():

                for iteration in range(ITERATIONS):
                    for prompt, negative in get_prompts():
                        print(f'*** Iteration {iteration + 1}: {prompt}')
                        tic2 = time.time()
                        images = generate_samples(model, sampler, prompt, negative, start_code)
                        for image in images:
                            name = save_image(image)
                            save_history(name, prompt, negative)
                            print(f'*** Saved image: {name}')
                        toc2 = time.time()

                        print(f'*** Synthesis time: {(toc2 - tic2):.2f}s')
                        counter += len(images)

    print(f'*** Total time: {(toc2 - tic1):.2f}s')
    print(f'*** Saved {counter} image(s) to {FOLDER} folder.')

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('*** User abort, goodbye.')
    except FileNotFoundError as e:
        print(f'*** {e}')
