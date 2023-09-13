import hashlib
import logging
import os
import sys
from contextlib import closing
from dataclasses import dataclass, field
from os.path import join
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from utility import masking, images

base_dir = os.getcwd()
sys.path.append(base_dir)
from configs.model_config import ModelPathConfig
from stable_diffusion.sampler.ddim import DDIMSampler
from stable_diffusion.sampler.ddpm import DDPMSampler
from stable_diffusion.sampler.diffusion import DiffusionSampler
from stable_diffusion.latent_diffusion import LatentDiffusion
from stable_diffusion.utils_backend import get_device, torch_gc
from stable_diffusion.utils_image import to_pil
from stable_diffusion import StableDiffusion
from stable_diffusion.model_paths import (SDconfigs)

# NOTE: It's just for the prompt embedder. Later refactor
import ga

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
opts.save_init_img = True
opts.img2img_color_correction = False
opts.img2img_background_color = '#ffffff'
opts.initial_noise_multiplier = 1.0
opts.outdir_init_images = 'output/init-images'
opts.sd_vae_encode_method = 'Full'

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


@dataclass(repr=False)
class StableDiffusionProcessing:
    sd_model: object = None
    outpath_samples: str = None
    prompt: str = ""
    prompt_for_display: str = None
    negative_prompt: str = ""
    styles: list[str] = None
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
    extra_generation_params: dict[str, Any] = None
    overlay_images: list = None

    cached_uc = [None, None]
    cached_c = [None, None]

    sampler: DiffusionSampler = field(default=None, init=False)

    c: tuple = field(default=None, init=False)
    uc: tuple = field(default=None, init=False)

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

    def prompt_embedding_vectors(self, prompt_array):
        embedded_prompts = ga.clip_text_get_prompt_embedding(self.config, prompts=prompt_array)
        embedded_prompts.to("cpu")
        return embedded_prompts

    def __post_init__(self):
        # NOTE: Initializing stable diffusion
        self.sd = StableDiffusion(device=self.device, n_steps=self.n_steps)
        self.config = ModelPathConfig()
        self.sd.quick_initialize().load_autoencoder(self.config.get_model(SDconfigs.VAE)).load_decoder(
            self.config.get_model(SDconfigs.VAE_DECODER))
        self.sd.model.load_unet(self.config.get_model(SDconfigs.UNET))
        self.sd.initialize_latent_diffusion(path='input/model/sd/v1-5-pruned-emaonly/v1-5-pruned-emaonly.safetensors',
                                            force_submodels_init=True)
        self.model = self.sd.model

        if self.styles is None:
            self.styles = []

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

        embedded_prompts = self.prompt_embedding_vectors(prompt_array=self.all_prompts)
        embedded_prompts_cpu = embedded_prompts.to("cpu")
        embedded_prompts_list = embedded_prompts_cpu.detach().numpy()

        prompt_embedding = torch.tensor(embedded_prompts_list[0], dtype=torch.float32)
        prompt_embedding = prompt_embedding.view(1, 77, 768).to(DEVICE)

        self.uc = self.prompt_embedding_vectors([""])[0]
        self.c = prompt_embedding


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

    # def images_tensor_to_samples(self,image, approximation=None, model=None):
    #     '''image[0, 1] -> latent'''
    #     if approximation is None:
    #         approximation = approximation_indexes.get(opts.sd_vae_encode_method, 0)
    #
    #     if approximation == 3:
    #         image = image.to(self.device, self.devices.dtype)
    #         x_latent = sd_vae_taesd.encoder_model()(image)
    #     else:
    #         if model is None:
    #             model = shared.sd_model
    #         model.first_stage_model.to(devices.dtype_vae)
    #
    #         image = image.to(shared.device, dtype=devices.dtype_vae)
    #         image = image * 2 - 1
    #         if len(image) > 1:
    #             x_latent = torch.stack([
    #                 model.get_first_stage_encoding(
    #                     model.encode_first_stage(torch.unsqueeze(img, 0))
    #                 )[0]
    #                 for img in image
    #             ])
    #         else:
    #             x_latent = model.get_first_stage_encoding(model.encode_first_stage(image))
    #
    #     return x_latent

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
                self.init_img_hash = hashlib.md5(img.tobytes()).hexdigest()
                img.save(join(opts.outdir_init_images, f"{self.init_img_hash}.png"))

            image = images.flatten(img, opts.img2img_background_color)

            if crop_region is None and self.resize_mode != 3:
                image = images.resize_image(self.resize_mode, image, self.width, self.height)
                # save image in storage
                image.save(join(opts.outdir_init_images, f"{self.init_img_hash}_resized.png"))

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
        x = create_random_tensors(shape=(1, 4, 64, 64))

        # mask = torch.zeros_like(x, device=self.device)
        # mask[:, :, mask.shape[2] // 2:, :] = 1.

        mask_np = np.zeros((64, 64), dtype=np.uint8)
        square_size = 8
        for i in range(0, 64, square_size * 2):
            for j in range(0, 64, square_size * 2):
                mask_np[i:i + square_size, j:j + square_size] = 1

        mask_np = np.tile(mask_np, (4, 1, 1))

        mask = torch.tensor(mask_np, dtype=torch.float32, device=DEVICE)
        mask = mask.unsqueeze(0)  # Adding batch dimension

        orig_noise = torch.randn(self.init_latent.shape, device=DEVICE)

        t_start = 35
        uncond_scale = 0.1
        x = self.sampler.q_sample(self.init_latent, t_start, noise=orig_noise)

        samples = self.sampler.paint(x=x,
                                     orig=self.init_latent,
                                     t_start=t_start,
                                     cond=conditioning,
                                     orig_noise=orig_noise,
                                     uncond_scale=uncond_scale,
                                     uncond_cond=unconditional_conditioning,
                                     mask=mask
                                     )
        torch_gc()

        return samples


def process_images(p: StableDiffusionProcessing):
    if isinstance(p.prompt, list):
        assert (len(p.prompt) > 0)
    else:
        assert p.prompt is not None

    torch_gc()

    seed = 123
    subseed = 456

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

            if len(p.prompts) == 0:
                break

            samples_ddim = p.sample(conditioning=p.c, unconditional_conditioning=p.uc, seeds=p.seeds,
                                    subseeds=p.subseeds, subseed_strength=p.subseed_strength, prompts=p.prompts)

            images = p.sd.get_image_from_latent(samples_ddim)
            pil_image = to_pil(images[0])
            pil_image.save('output/inpainting/result.png')

            del samples_ddim

            # with torch.cuda.device('cpu'):
            #     torch.cuda.empty_cache()
            #     torch.cuda.ipc_collect()


def create_binary_mask(image):
    if image.mode == 'RGBA' and image.getextrema()[-1] != (255, 255):
        image = image.split()[-1].convert("L").point(lambda x: 255 if x > 128 else 0)
    else:
        image = image.convert('L')
    return image


def img2img(prompt: str, negative_prompt: str, sampler_name: str, batch_size: int, n_iter: int, steps: int,
            cfg_scale: float, width: int, height: int, mask_blur: int, inpainting_fill: int, init_img):
    image = init_img

    p = StableDiffusionProcessingImg2Img(
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        prompt=prompt,
        negative_prompt=negative_prompt,
        styles=[],
        sampler_name=sampler_name,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        init_images=[image],
        mask=create_binary_mask(init_maks),
        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        resize_mode=0,
        denoising_strength=0.75,
        image_cfg_scale=1.5,
        inpaint_full_res_padding=32,
        inpainting_mask_invert=0,
    )

    with closing(p):
        process_images(p)


init_image = Image.open("test.png")
init_maks = Image.open("mask.png")


img2img(prompt="A cat",
        negative_prompt="",
        sampler_name="ddim",
        batch_size=1,
        n_iter=1,
        steps=20,
        cfg_scale=7,
        width=512,
        height=512,
        mask_blur=4,
        inpainting_fill=4,
        init_img=init_image)
