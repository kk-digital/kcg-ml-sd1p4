import argparse
import hashlib
import logging
import os
import random
import sys
import time
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from os.path import join
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from blendmodes.blend import blendLayers
from blendmodes.blendtype import BlendType
from skimage import exposure

base_dir = os.getcwd()
sys.path.append(base_dir)
from utility import masking, images, rng, prompt_parser
from utility.rng import ImageRNG
from utility.utils_logger import logger
from configs.model_config import ModelPathConfig
from stable_diffusion.sampler.ddim import DDIMSampler
from stable_diffusion.sampler.ddpm import DDPMSampler
from stable_diffusion.sampler.diffusion import DiffusionSampler
from stable_diffusion.latent_diffusion import LatentDiffusion
from stable_diffusion.utils_backend import get_device, torch_gc, without_autocast, get_autocast
from stable_diffusion import StableDiffusion, CLIPTextEmbedder
from stable_diffusion.model_paths import (SDconfigs, CLIPconfigs)

# NOTE: It's just for the prompt embedder. Later refactor

output_dir = join(base_dir, 'output', 'inpainting')
os.makedirs(output_dir, exist_ok=True)


class Options:
    outdir_samples: str
    save_init_img: bool
    img2img_color_correction: bool
    img2img_background_color: str


# some of those options should not be changed at all because they would break the model, so I removed them from options.
opt_C = 4
opt_f = 8
opts = Options()

opts.outdir_samples = output_dir
opts.save_init_img = False
opts.img2img_color_correction = False
opts.img2img_background_color = '#ffffff'
opts.initial_noise_multiplier = 1.0
opts.outdir_init_images = 'output/init-images'
opts.sd_vae_encode_method = 'Full'
opts.sd_vae_decode_method = 'Full'
opts.CLIP_stop_at_last_layers = 1
opts.sdxl_crop_left = 0
opts.sdxl_crop_top = 0
opts.use_old_scheduling = False

approximation_indexes = {"Full": 0, "Approx NN": 1, "Approx cheap": 2, "TAESD": 3}

DEVICE = get_device()
PROMPT_STYLES = None


def flatten(img, bgcolor):
    if img.mode == "RGBA":
        background = Image.new('RGBA', img.size, bgcolor)
        background.paste(img, mask=img)
        img = background

    return img.convert('RGB')


def create_binary_mask(image):
    if image.mode == 'RGBA' and image.getextrema()[-1] != (255, 255):
        image = image.split()[-1].convert("L").point(lambda x: 255 if x > 128 else 0)
    else:
        image = image.convert('L')
    return image


def create_random_tensors(shape=(1, 4, 64, 64), low=0.0, high=1.0, device=DEVICE, requires_grad=False):
    random_tensor = torch.tensor(np.random.uniform(low=low, high=high, size=shape), dtype=torch.float32, device=device,
                                 requires_grad=requires_grad)
    return random_tensor


def setup_color_correction(image):
    logging.info("Calibrating color correction.")
    correction_target = cv2.cvtColor(np.asarray(image.copy()), cv2.COLOR_RGB2LAB)
    return correction_target


def get_fixed_seed(seed):
    if seed == '' or seed is None:
        seed = -1
    elif isinstance(seed, str):
        try:
            seed = int(seed)
        except Exception:
            seed = -1

    if seed == -1:
        return int(random.randrange(4294967294))

    return seed


class DecodedSamples(list):
    already_decoded = True


def samples_to_images_tensor(sample, approximation=None, model=None):
    """Transforms 4-channel latent space images into 3-channel RGB image tensors, with values in range [-1, 1]."""

    with without_autocast():  # fixes an issue with unstable VAEs that are flaky even in fp32
        x_sample = model.autoencoder_decode(sample.to(torch.float32))

    return x_sample


def decode_first_stage(model, x):
    x = x.to(torch.float32)
    approx_index = approximation_indexes.get(opts.sd_vae_decode_method, 0)
    return samples_to_images_tensor(x, approx_index, model)


def decode_latent_batch(model, batch, target_device=None, check_for_nans=False):
    samples = DecodedSamples()

    for i in range(batch.shape[0]):
        sample = decode_first_stage(model, batch[i:i + 1])[0]

        if target_device is not None:
            sample = sample.to(target_device)

        samples.append(sample)

    return samples


@dataclass(repr=False)
class StableDiffusionProcessing:
    sd_model: object = None
    outpath: str = None
    prompt: str = ""
    prompt_for_display: str = None
    negative_prompt: str = ""
    styles: list = None
    seed: int = -1
    subseed: int = -1
    subseed_strength: float = 0
    seed_resize_from_h: int = -1
    seed_resize_from_w: int = -1
    seed_enable_extras: bool = True
    sampler_name: str = None
    batch_size: int = 1
    n_iter: int = 1
    steps: int = 50
    cfg_scale: float = 7.0
    width: int = 512
    height: int = 512
    extra_generation_params: dict = None
    overlay_images: list = None

    cached_uc = [None, None]
    cached_c = [None, None]

    sampler: DiffusionSampler = field(default=None, init=False)

    c: tuple = field(default=None, init=False)
    uc: tuple = field(default=None, init=False)

    rng: ImageRNG = field(default=None, init=False)
    color_corrections: list = field(default=None, init=False)

    all_prompts: list = field(default=None, init=False)
    all_negative_prompts: list = field(default=None, init=False)
    all_seeds: list = field(default=None, init=False)
    all_subseeds: list = field(default=None, init=False)
    iteration: int = field(default=0, init=False)
    main_prompt: str = field(default=None, init=False)
    main_negative_prompt: str = field(default=None, init=False)

    prompts: list = field(default=None, init=False)
    negative_prompts: list = field(default=None, init=False)
    seeds: list = field(default=None, init=False)
    subseeds: list = field(default=None, init=False)

    sd: StableDiffusion = None
    config: ModelPathConfig = None
    model: LatentDiffusion = None
    n_steps: int = 50
    ddim_eta: float = 0.0
    device = get_device()

    clip_text_embedder = CLIPTextEmbedder(device=get_device())

    def prompt_embedding_vectors(self, prompt_array):
        embedded_prompts = []
        for prompt in prompt_array:
            prompt_embedding = self.clip_text_embedder.forward(prompt)
            embedded_prompts.append(prompt_embedding)

        embedded_prompts = torch.stack(embedded_prompts)

        return embedded_prompts

    def __post_init__(self):
        if self.sd is None:
            # NOTE: Initializing stable diffusion
            self.sd = StableDiffusion(device=self.device, n_steps=self.n_steps)
            self.config = ModelPathConfig()
            self.sd.quick_initialize().load_autoencoder(self.config.get_model(SDconfigs.VAE)).load_decoder(
                self.config.get_model(SDconfigs.VAE_DECODER))
            self.sd.model.load_unet(self.config.get_model(SDconfigs.UNET))
            self.sd.initialize_latent_diffusion(
                path='input/model/sd/v1-5-pruned-emaonly/v1-5-pruned-emaonly.safetensors',
                force_submodels_init=True)
            self.model = self.sd.model

        if self.styles is None:
            self.styles = []

        self.clip_text_embedder.load_submodels(
            tokenizer_path=self.config.get_model_folder_path(CLIPconfigs.TXT_EMB_TOKENIZER),
            transformer_path=self.config.get_model_folder_path(CLIPconfigs.TXT_EMB_TEXT_MODEL)
        )

        self.cached_uc = StableDiffusionProcessing.cached_uc
        self.cached_c = StableDiffusionProcessing.cached_c
        self.extra_generation_params = self.extra_generation_params or {}

    def init(self, all_prompts, all_seeds, all_subseeds):
        pass

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        raise NotImplementedError()

    def close(self):
        self.sampler = None
        self.c = None
        self.uc = None
        StableDiffusionProcessing.cached_c = [None, None]
        StableDiffusionProcessing.cached_uc = [None, None]

        self.clip_text_embedder.to("cpu")
        torch.cuda.empty_cache()

    def setup_prompts(self):
        if isinstance(self.prompt, list):
            self.all_prompts = self.prompt
        elif isinstance(self.negative_prompt, list):
            self.all_prompts = [self.prompt] * len(self.negative_prompt)
        else:
            self.all_prompts = self.batch_size * self.n_iter * [self.prompt]

        if isinstance(self.negative_prompt, list):
            self.all_negative_prompts = self.negative_prompt
        else:
            self.all_negative_prompts = [self.negative_prompt] * len(self.all_prompts)

        if len(self.all_prompts) != len(self.all_negative_prompts):
            raise RuntimeError(
                f"Received a different number of prompts ({len(self.all_prompts)}) and negative prompts ({len(self.all_negative_prompts)})")

        self.main_prompt = self.all_prompts[0]
        self.main_negative_prompt = self.all_negative_prompts[0]

    def cached_params(self, required_prompts, steps, extra_network_data, hires_steps=None, use_old_scheduling=False):
        """Returns parameters that invalidate the cond cache if changed"""

        return (
            required_prompts,
            steps,
            hires_steps,
            use_old_scheduling,
            opts.CLIP_stop_at_last_layers,
            '',
            extra_network_data,
            opts.sdxl_crop_left,
            opts.sdxl_crop_top,
            self.width,
            self.height,
        )

    def get_conds_with_caching(self, function, required_prompts, steps, caches, extra_network_data, hires_steps=None):
        """
        Returns the result of calling function(shared.sd_model, required_prompts, steps)
        using a cache to store the result if the same arguments have been used before.

        cache is an array containing two elements. The first element is a tuple
        representing the previously used arguments, or None if no arguments
        have been used before. The second element is where the previously
        computed result is stored.

        caches is a list with items described above.
        """

        cached_params = self.cached_params(required_prompts, steps, extra_network_data, hires_steps,
                                           opts.use_old_scheduling)

        for cache in caches:
            if cache[0] is not None and cached_params == cache[0]:
                return cache[1]

        cache = caches[0]

        with get_autocast():
            cache[1] = function(self.model, required_prompts, steps, hires_steps, opts.use_old_scheduling)

        cache[0] = cached_params
        return cache[1]

    def setup_conds(self):
        prompts = prompt_parser.SdConditioning(self.prompts, width=self.width, height=self.height)
        negative_prompts = prompt_parser.SdConditioning(self.negative_prompts, width=self.width, height=self.height,
                                                        is_negative_prompt=True)

        self.uc = self.prompt_embedding_vectors(negative_prompts)[0]
        self.c = self.prompt_embedding_vectors(prompts)[0]


@dataclass(repr=False)
class StableDiffusionProcessingImg2Img(StableDiffusionProcessing):
    init_images: list = None
    resize_mode: int = 0
    denoising_strength: float = 0.75
    image_cfg_scale: float = None
    mask: Any = None
    mask_blur_x: int = 4
    mask_blur_y: int = 4
    mask_blur: int = None
    inpainting_fill: int = 0
    inpaint_full_res: bool = True
    inpaint_full_res_padding: int = 0
    inpainting_mask_invert: int = 0
    initial_noise_multiplier: float = None
    latent_mask: Image = None

    image_mask: Any = field(default=None, init=False)

    nmask: torch.Tensor = field(default=None, init=False)
    image_conditioning: torch.Tensor = field(default=None, init=False)
    init_img_hash: str = field(default=None, init=False)
    mask_for_overlay: Image = field(default=None, init=False)
    init_latent: torch.Tensor = field(default=None, init=False)
    device = get_device()

    def __post_init__(self):
        super().__post_init__()

        self.image_mask = self.mask
        self.mask = None
        self.initial_noise_multiplier = opts.initial_noise_multiplier if self.initial_noise_multiplier is None else self.initial_noise_multiplier

    @property
    def mask_blur(self):
        if self.mask_blur_x == self.mask_blur_y:
            return self.mask_blur_x
        return None

    @mask_blur.setter
    def mask_blur(self, value):
        if isinstance(value, int):
            self.mask_blur_x = value
            self.mask_blur_y = value

    def init(self, all_prompts, all_seeds, all_subseeds):
        self.image_cfg_scale: float = None

        if self.sampler_name == 'ddim':
            self.sampler = DDIMSampler(self.model,
                                       n_steps=self.n_steps,
                                       ddim_eta=self.ddim_eta)
        elif self.sampler_name == 'ddpm':
            self.sampler = DDPMSampler(self.model)

        crop_region = None

        image_mask = self.image_mask

        if image_mask is not None:
            # image_mask is passed in as RGBA by Gradio to support alpha masks,
            # but we still want to support binary masks.
            image_mask = create_binary_mask(image_mask)

            if self.inpainting_mask_invert:
                image_mask = ImageOps.invert(image_mask)

            if self.mask_blur_x > 0:
                np_mask = np.array(image_mask)
                kernel_size = 2 * int(2.5 * self.mask_blur_x + 0.5) + 1
                np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), self.mask_blur_x)
                image_mask = Image.fromarray(np_mask)

            if self.mask_blur_y > 0:
                np_mask = np.array(image_mask)
                kernel_size = 2 * int(2.5 * self.mask_blur_y + 0.5) + 1
                np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), self.mask_blur_y)
                image_mask = Image.fromarray(np_mask)

            if self.inpaint_full_res:
                self.mask_for_overlay = image_mask
                mask = image_mask.convert('L')
                crop_region = masking.get_crop_region(np.array(mask), self.inpaint_full_res_padding)
                crop_region = masking.expand_crop_region(crop_region, self.width, self.height, mask.width, mask.height)
                x1, y1, x2, y2 = crop_region

                mask = mask.crop(crop_region)
                image_mask = images.resize_image(2, mask, self.width, self.height)
                self.paste_to = (x1, y1, x2 - x1, y2 - y1)
            else:
                image_mask = images.resize_image(self.resize_mode, image_mask, self.width, self.height)
                np_mask = np.array(image_mask)
                np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.uint8)
                self.mask_for_overlay = Image.fromarray(np_mask)

            self.overlay_images = []

        latent_mask = self.latent_mask if self.latent_mask is not None else image_mask

        add_color_corrections = opts.img2img_color_correction and self.color_corrections is None
        if add_color_corrections:
            self.color_corrections = []
        imgs = []
        for img in self.init_images:

            # Save init image
            if opts.save_init_img:
                if not os.path.exists(opts.outdir_init_images):
                    os.makedirs(opts.outdir_init_images)
                self.init_img_hash = hashlib.md5(img.tobytes()).hexdigest()
                img.save(join(opts.outdir_init_images, f"{self.init_img_hash}.png"))

            image = images.flatten(img, opts.img2img_background_color)

            if crop_region is None and self.resize_mode != 3:
                image = images.resize_image(self.resize_mode, image, self.width, self.height)

            if image_mask is not None:
                image_masked = Image.new('RGBa', (image.width, image.height))
                image_masked.paste(image.convert("RGBA").convert("RGBa"),
                                   mask=ImageOps.invert(self.mask_for_overlay.convert('L')))

                self.overlay_images.append(image_masked.convert('RGBA'))

            # crop_region is not None if we are doing inpaint full res
            if crop_region is not None:
                image = image.crop(crop_region)
                image = images.resize_image(2, image, self.width, self.height)

            if image_mask is not None:
                if self.inpainting_fill != 1:
                    image = masking.fill(image, latent_mask)

            if add_color_corrections:
                self.color_corrections.append(setup_color_correction(image))

            image = np.array(image).astype(np.float32) / 255.0
            image = np.moveaxis(image, 2, 0)

            imgs.append(image)

        if len(imgs) == 1:
            batch_images = np.expand_dims(imgs[0], axis=0).repeat(self.batch_size, axis=0)
            if self.overlay_images is not None:
                self.overlay_images = self.overlay_images * self.batch_size

            if self.color_corrections is not None and len(self.color_corrections) == 1:
                self.color_corrections = self.color_corrections * self.batch_size

        elif len(imgs) <= self.batch_size:
            self.batch_size = len(imgs)
            batch_images = np.array(imgs)
        else:
            raise RuntimeError(f"bad number of images passed: {len(imgs)}; expecting {self.batch_size} or less")

        image = torch.from_numpy(batch_images)
        image = image.to(self.device, dtype=torch.float32)

        if opts.sd_vae_encode_method != 'Full':
            self.extra_generation_params['VAE Encoder'] = opts.sd_vae_encode_method

        # self.init_latent = images_tensor_to_samples(image, approximation_indexes.get(opts.sd_vae_encode_method),
        #                                             self.sd_model)
        self.init_latent = self.model.autoencoder_encode(image)
        torch_gc()

        if self.resize_mode == 3:
            self.init_latent = torch.nn.functional.interpolate(self.init_latent,
                                                               size=(self.height // opt_f, self.width // opt_f),
                                                               mode="bilinear")

        if image_mask is not None:
            init_mask = latent_mask
            latmask = init_mask.convert('RGB').resize((self.init_latent.shape[3], self.init_latent.shape[2]))
            latmask = np.moveaxis(np.array(latmask, dtype=np.float32), 2, 0) / 255
            latmask = latmask[0]
            latmask = np.around(latmask)
            latmask = np.tile(latmask[None], (4, 1, 1))

            self.mask = torch.asarray(1.0 - latmask).to(self.device).type(torch.float32)
            self.nmask = torch.asarray(latmask).to(self.device).type(torch.float32)

            # this needs to be fixed to be done in sample() using actual seeds for batches
            if self.inpainting_fill == 2:
                self.init_latent = self.init_latent * self.mask + create_random_tensors(self.init_latent.shape[1:],
                                                                                        all_seeds[
                                                                                        0:self.init_latent.shape[
                                                                                            0]]) * self.nmask
            elif self.inpainting_fill == 3:
                self.init_latent = self.init_latent * self.mask

        self.image_conditioning = self.init_latent.new_zeros(self.init_latent.shape[0], 5, 1, 1)

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        orig_noise = self.rng.next()
        x = self.rng.next()

        if self.initial_noise_multiplier != 1.0:
            self.extra_generation_params["Noise multiplier"] = self.initial_noise_multiplier
            x *= self.initial_noise_multiplier

        # orig_noise = torch.randn(self.init_latent.shape, device=self.device)

        t_start = 35
        uncond_scale = 100

        samples = self.sampler.paint(x=x,
                                     orig=self.init_latent,
                                     t_start=t_start,
                                     cond=conditioning,
                                     orig_noise=orig_noise,
                                     uncond_scale=uncond_scale,
                                     uncond_cond=unconditional_conditioning,
                                     mask=self.mask,
                                     )
        if self.mask is not None:
            samples = samples * self.nmask + self.init_latent * self.mask

        torch_gc()

        return samples


def apply_color_correction(correction, original_image):
    logging.info("Applying color correction.")
    image = Image.fromarray(cv2.cvtColor(exposure.match_histograms(
        cv2.cvtColor(
            np.asarray(original_image),
            cv2.COLOR_RGB2LAB
        ),
        correction,
        channel_axis=2
    ), cv2.COLOR_LAB2RGB).astype("uint8"))

    image = blendLayers(image, original_image, BlendType.LUMINOSITY)

    return image.convert('RGB')


def apply_overlay(image, paste_loc, index, overlays):
    if overlays is None or index >= len(overlays):
        return image

    overlay = overlays[index]

    if paste_loc is not None:
        x, y, w, h = paste_loc
        base_image = Image.new('RGBA', (overlay.width, overlay.height))
        image = images.resize_image(1, image, w, h)
        base_image.paste(image, (x, y))
        image = base_image

    image = image.convert('RGBA')
    image.alpha_composite(overlay)
    image = image.convert('RGB')

    return image


def process_images(p: StableDiffusionProcessingImg2Img):
    if isinstance(p.prompt, list):
        assert (len(p.prompt) > 0)
    else:
        assert p.prompt is not None

    torch_gc()

    seed = get_fixed_seed(p.seed)
    subseed = get_fixed_seed(p.subseed)

    # Check if is correct
    p.setup_prompts()

    if isinstance(seed, list):
        p.all_seeds = seed
    else:
        p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]

    if isinstance(subseed, list):
        p.all_subseeds = subseed
    else:
        p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]

    with torch.no_grad():
        with torch.autocast("cpu"):
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)

        for n in range(p.n_iter):
            p.iteration = n
            p.prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            p.negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            p.seeds = p.all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
            p.subseeds = p.all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

            # May we need to configure this part to get the propaly conds
            p.setup_conds()

            p.rng = rng.ImageRNG((opt_C, p.height // opt_f, p.width // opt_f), p.seeds, subseeds=p.subseeds,
                                 subseed_strength=p.subseed_strength, seed_resize_from_h=p.seed_resize_from_h,
                                 seed_resize_from_w=p.seed_resize_from_w)
            if len(p.prompts) == 0:
                break

            with without_autocast():
                samples_ddim = p.sample(conditioning=p.c, unconditional_conditioning=p.uc, seeds=p.seeds,
                                        subseeds=p.subseeds, subseed_strength=p.subseed_strength, prompts=p.prompts)

            if getattr(samples_ddim, 'already_decoded', False):
                x_samples_ddim = samples_ddim
            else:
                if opts.sd_vae_decode_method != 'Full':
                    p.extra_generation_params['VAE Decoder'] = opts.sd_vae_decode_method

                x_samples_ddim = decode_latent_batch(p.model, samples_ddim, target_device=torch.device('cpu'),
                                                     check_for_nans=True)

            x_samples_ddim = torch.stack(x_samples_ddim).float()
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            del samples_ddim

            for i, x_sample in enumerate(x_samples_ddim):
                p.batch_index = i

                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)

                image = Image.fromarray(x_sample)

                image = apply_overlay(image, p.paste_to, i, p.overlay_images)
                image.save(join(p.outpath, f"{int(time.time())}.png"))

            del x_samples_ddim
            torch_gc()


def create_binary_mask(image):
    if image.mode == 'RGBA' and image.getextrema()[-1] != (255, 255):
        image = image.split()[-1].convert("L").point(lambda x: 255 if x > 128 else 0)
    else:
        image = image.convert('L')
    return image


def get_model(device, n_steps):
    # NOTE: Initializing stable diffusion
    sd = StableDiffusion(device=device, n_steps=n_steps)
    config = ModelPathConfig()
    sd.quick_initialize().load_autoencoder(config.get_model(SDconfigs.VAE)).load_decoder(
        config.get_model(SDconfigs.VAE_DECODER))
    sd.model.load_unet(config.get_model(SDconfigs.UNET))
    sd.initialize_latent_diffusion(path='input/model/sd/v1-5-pruned-emaonly/v1-5-pruned-emaonly.safetensors',
                                   force_submodels_init=True)
    model = sd.model

    return sd, config, model


def img2img(prompt: str, negative_prompt: str, sampler_name: str, batch_size: int, n_iter: int, steps: int,
            cfg_scale: float, width: int, height: int, mask_blur: int, inpainting_fill: int,
            outpath, styles, init_images, mask, resize_mode, denoising_strength,
            image_cfg_scale, inpaint_full_res_padding, inpainting_mask_invert, sd=None, config=None, model=None):
    p = StableDiffusionProcessingImg2Img(
        outpath=outpath,
        prompt=prompt,
        negative_prompt=negative_prompt,
        styles=styles,
        sampler_name=sampler_name,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        init_images=init_images,
        mask=create_binary_mask(mask),
        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        resize_mode=resize_mode,
        denoising_strength=denoising_strength,
        image_cfg_scale=image_cfg_scale,
        inpaint_full_res_padding=inpaint_full_res_padding,
        inpainting_mask_invert=inpainting_mask_invert,
        sd=sd,
        config=config,
        model=model
    )

    with closing(p):
        process_images(p)

def parse_args():
    parser = argparse.ArgumentParser(description="Call img2img with specified parameters.")

    # Required parameters
    parser.add_argument("--prompt", type=str, help="Input prompt")
    parser.add_argument("--init_img", type=str, help="Path to the initial image")
    parser.add_argument("--init_mask", type=str, help="Path to the initial mask")

    # Optional parameters with default values
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--sampler_name", type=str, default="ddim", help="Sampler name")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--n_iter", type=int, default=1, help="Number of iterations")
    parser.add_argument("--steps", type=int, default=20, help="Steps")
    parser.add_argument("--cfg_scale", type=float, default=7.0, help="Config scale")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--mask_blur", type=int, default=4, help="Mask blur value")
    parser.add_argument("--inpainting_fill", type=int, default=1, help="Inpainting fill value")

    # Additional parameters for StableDiffusionProcessingImg2Img
    # Add default values as needed
    parser.add_argument("--outpath", type=str, default=f"output/inpainting/{datetime.now().strftime('%m-%d-%Y')}",
                        help="Output path for samples")
    parser.add_argument("--styles", nargs="*", default=[], help="Styles list")
    parser.add_argument("--resize_mode", type=int, default=0, help="Resize mode")
    parser.add_argument("--denoising_strength", type=float, default=0.75, help="Denoising strength")
    parser.add_argument("--image_cfg_scale", type=float, default=1.5, help="Image config scale")
    parser.add_argument("--inpaint_full_res_padding", type=int, default=32, help="Inpaint full resolution padding")
    parser.add_argument("--inpainting_mask_invert", type=int, default=0, help="Inpainting mask invert value")

    return parser.parse_args()


def main():
    args = parse_args()

    init_image = Image.open(args.init_img)
    init_mask = Image.open(args.init_mask)

    # Displaying the parameters using the logger
    logger.info("Parameters for img2img:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")

    # Create the output directory if it does not exist
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)

    img2img(prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            sampler_name=args.sampler_name,
            batch_size=args.batch_size,
            n_iter=args.n_iter,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            width=args.width,
            height=args.height,
            mask_blur=args.mask_blur,
            inpainting_fill=args.inpainting_fill,
            outpath=args.outpath,
            styles=args.styles,
            init_images=[init_image],
            mask=init_mask,
            resize_mode=args.resize_mode,
            denoising_strength=args.denoising_strength,
            image_cfg_scale=args.image_cfg_scale,
            inpaint_full_res_padding=args.inpaint_full_res_padding,
            inpainting_mask_invert=args.inpainting_mask_invert
            )


if __name__ == "__main__":
    main()