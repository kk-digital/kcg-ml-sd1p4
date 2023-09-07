import os
import torch
import hashlib
from os.path import join
import sys
import json
import cv2
from contextlib import closing
from pathlib import Path

import numpy as np
from dataclasses import dataclass, field
from typing import Any
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, UnidentifiedImageError
# import gradio as gr

# NOTE: Our libraries
base_dir = os.getcwd()
sys.path.append(base_dir)
from configs.model_config import ModelPathConfig
from stable_diffusion.sampler.ddim import DDIMSampler
from stable_diffusion.sampler.ddpm import DDPMSampler
from stable_diffusion.sampler.diffusion import DiffusionSampler
from stable_diffusion.latent_diffusion import LatentDiffusion
from stable_diffusion.utils_model import initialize_latent_diffusion
from stable_diffusion.utils_backend import get_device
from stable_diffusion.utils_image import to_pil
from stable_diffusion import StableDiffusion
from stable_diffusion.model_paths import (SDconfigs)

# NOTE: It's just for the prompt embedder. Later refactor
import ga

# from modules import images as imgutil
# from modules.generation_parameters_copypaste import create_override_settings_dict, parse_generation_parameters
# from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images
# from modules.shared import opts, state
# import modules.shared as shared
# import modules.processing as processing
# from modules.ui import plaintext_to_html
# import modules.scripts

# opts = None
output_dir = join(base_dir, 'output', 'inpainting')
os.makedirs(output_dir, exist_ok=True)

class Options:
    outdir_samples: str
    # initial_noise_multiplier: float
    save_init_img: bool
    img2img_color_correction: bool
    img2img_background_color: str

opts = Options()

opts.outdir_samples = output_dir
# opts.initial_noise_multiplier = 1.0
opts.save_init_img = False
opts.img2img_color_correction = False
opts.img2img_background_color = '#ffffff'

# NOTE: I think state is used for having a job queue for the web app
# state = None
# NOTE: Init SD_MODEL
SD_MODEL = None
DEVICE = get_device()
PROMPT_STYLES = None
# TOTAL_TQDM = None

def flatten(img, bgcolor):
    """replaces transparency with bgcolor (example: "#ffffff"), returning an RGB mode image with no transparency"""

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
    random_tensor = torch.tensor(np.random.uniform(low=low, high=high, size=shape), dtype=torch.float32, device=device, requires_grad=requires_grad)
    return random_tensor

@dataclass(repr=False)
class StableDiffusionProcessing:
    sd_model: object = None
    outpath_samples: str = None
    # outpath_grids: str = None
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
    restore_faces: bool = None
    tiling: bool = None
    do_not_save_samples: bool = False
    do_not_save_grid: bool = False
    extra_generation_params: dict[str, Any] = None
    overlay_images: list = None
    eta: float = None
    do_not_reload_embeddings: bool = False
    denoising_strength: float = 0
    ddim_discretize: str = None
    # s_min_uncond: float = None
    # s_churn: float = None
    # s_tmax: float = None
    # s_tmin: float = None
    # s_noise: float = None
    # override_settings: dict[str, Any] = None
    # override_settings_restore_afterwards: bool = True
    sampler_index: int = None
    refiner_checkpoint: str = None
    refiner_switch_at: float = None
    token_merging_ratio = 0
    token_merging_ratio_hr = 0
    disable_extra_networks: bool = False

    # scripts_value: scripts.ScriptRunner = field(default=None, init=False)
    # script_args_value: list = field(default=None, init=False)
    # scripts_setup_complete: bool = field(default=False, init=False)

    cached_uc = [None, None]
    cached_c = [None, None]

    comments: dict = None
    # NOTE: We'll need to setup our sampler later. Trickier, because method calls need to match.
    # sampler: sd_samplers_common.Sampler | None = field(default=None, init=False)
    sampler: DiffusionSampler = field(default=None, init=False)
    is_using_inpainting_conditioning: bool = field(default=False, init=False)
    paste_to: tuple | None = field(default=None, init=False)

    is_hr_pass: bool = field(default=False, init=False)

    c: tuple = field(default=None, init=False)
    uc: tuple = field(default=None, init=False)

    # rng: rng.ImageRNG | None = field(default=None, init=False)
    step_multiplier: int = field(default=1, init=False)
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
    extra_network_data: dict = field(default=None, init=False)

    user: str = field(default=None, init=False)

    sd_model_name: str = field(default=None, init=False)
    sd_model_hash: str = field(default=None, init=False)
    sd_vae_name: str = field(default=None, init=False)
    sd_vae_hash: str = field(default=None, init=False)

    is_api: bool = field(default=False, init=False)

    # NOTE: Originally not here
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
        if self.sampler_index is not None:
            print("sampler_index argument for StableDiffusionProcessing does not do anything; use sampler_name", file=sys.stderr)

        # NOTE: Initializing stable diffusion
        self.sd = StableDiffusion(device=self.device, n_steps=self.n_steps)
        self.config = ModelPathConfig()
        self.sd.quick_initialize().load_autoencoder(self.config.get_model(SDconfigs.VAE)).load_decoder(self.config.get_model(SDconfigs.VAE_DECODER))
        self.sd.model.load_unet(self.config.get_model(SDconfigs.UNET))
        self.sd.initialize_latent_diffusion(path='input/model/sd/v1-5-pruned-emaonly/v1-5-pruned-emaonly.safetensors', force_submodels_init=True)
        self.model = self.sd.model

        self.comments = {}

        if self.styles is None:
            self.styles = []

        self.sampler_noise_scheduler_override = None
        # self.s_min_uncond = self.s_min_uncond if self.s_min_uncond is not None else opts.s_min_uncond
        # self.s_churn = self.s_churn if self.s_churn is not None else opts.s_churn
        # self.s_tmin = self.s_tmin if self.s_tmin is not None else opts.s_tmin
        # self.s_tmax = (self.s_tmax if self.s_tmax is not None else opts.s_tmax) or float('inf')
        # self.s_noise = self.s_noise if self.s_noise is not None else opts.s_noise

        self.extra_generation_params = self.extra_generation_params or {}
        # self.override_settings = self.override_settings or {}
        # self.script_args = self.script_args or {}

        self.refiner_checkpoint_info = None

        if not self.seed_enable_extras:
            self.subseed = -1
            self.subseed_strength = 0
            self.seed_resize_from_h = 0
            self.seed_resize_from_w = 0

        self.cached_uc = StableDiffusionProcessing.cached_uc
        self.cached_c = StableDiffusionProcessing.cached_c

    @property
    def sd_model(self):
        # return shared.sd_model
        return SD_MODEL

    @sd_model.setter
    def sd_model(self, value):
        pass

    # @property
    # def scripts(self):
    #     return self.scripts_value

    # @scripts.setter
    # def scripts(self, value):
    #     self.scripts_value = value

    #     if self.scripts_value and self.script_args_value and not self.scripts_setup_complete:
    #         self.setup_scripts()

    # @property
    # def script_args(self):
    #     return self.script_args_value

    # @script_args.setter
    # def script_args(self, value):
    #     self.script_args_value = value

        # if self.scripts_value and self.script_args_value and not self.scripts_setup_complete:
        #     self.setup_scripts()

    # def setup_scripts(self):
    #     self.scripts_setup_complete = True

    #     self.scripts.setup_scrips(self, is_ui=not self.is_api)

    # def comment(self, text):
    #     self.comments[text] = 1

    # def txt2img_image_conditioning(self, x, width=None, height=None):
    #     self.is_using_inpainting_conditioning = self.sd_model.model.conditioning_key in {'hybrid', 'concat'}

    #     return txt2img_image_conditioning(self.sd_model, x, width or self.width, height or self.height)

    # def depth2img_image_conditioning(self, source_image):
    #     # Use the AddMiDaS helper to Format our source image to suit the MiDaS model
    #     transformer = AddMiDaS(model_type="dpt_hybrid")
    #     transformed = transformer({"jpg": rearrange(source_image[0], "c h w -> h w c")})
    #     # midas_in = torch.from_numpy(transformed["midas_in"][None, ...]).to(device=shared.device)
    #     midas_in = torch.from_numpy(transformed["midas_in"][None, ...]).to(device=DEVICE)
    #     midas_in = repeat(midas_in, "1 ... -> n ...", n=self.batch_size)

    #     conditioning_image = images_tensor_to_samples(source_image*0.5+0.5, approximation_indexes.get(opts.sd_vae_encode_method))
    #     conditioning = torch.nn.functional.interpolate(
    #         self.sd_model.depth_model(midas_in),
    #         size=conditioning_image.shape[2:],
    #         mode="bicubic",
    #         align_corners=False,
    #     )

    #     (depth_min, depth_max) = torch.aminmax(conditioning)
    #     conditioning = 2. * (conditioning - depth_min) / (depth_max - depth_min) - 1.
    #     return conditioning

    # def edit_image_conditioning(self, source_image):
    #     conditioning_image = images_tensor_to_samples(source_image*0.5+0.5, approximation_indexes.get(opts.sd_vae_encode_method))

    #     return conditioning_image

    # def unclip_image_conditioning(self, source_image):
    #     c_adm = self.sd_model.embedder(source_image)
    #     if self.sd_model.noise_augmentor is not None:
    #         noise_level = 0 # TODO: Allow other noise levels?
    #         c_adm, noise_level_emb = self.sd_model.noise_augmentor(c_adm, noise_level=repeat(torch.tensor([noise_level]).to(c_adm.device), '1 -> b', b=c_adm.shape[0]))
    #         c_adm = torch.cat((c_adm, noise_level_emb), 1)
    #     return c_adm

    def inpainting_image_conditioning(self, source_image, latent_image, image_mask=None):
        # self.is_using_inpainting_conditioning = True

        # Handle the different mask inputs
        if image_mask is not None:
            if torch.is_tensor(image_mask):
                conditioning_mask = image_mask
            else:
                conditioning_mask = np.array(image_mask.convert("L"))
                conditioning_mask = conditioning_mask.astype(np.float32) / 255.0
                conditioning_mask = torch.from_numpy(conditioning_mask[None, None])

                # Inpainting model uses a discretized mask as input, so we round to either 1.0 or 0.0
                conditioning_mask = torch.round(conditioning_mask)
        else:
            conditioning_mask = source_image.new_ones(1, 1, *source_image.shape[-2:])

        # Create another latent image, this time with a masked version of the original input.
        # Smoothly interpolate between the masked and unmasked latent conditioning image using a parameter.
        conditioning_mask = conditioning_mask.to(device=source_image.device, dtype=source_image.dtype)
        conditioning_image = torch.lerp(
            source_image,
            source_image * (1.0 - conditioning_mask),
            1.0
        )

        # Encode the new masked image using first stage of network.
        # conditioning_image = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(conditioning_image))
        conditioning_image = self.model.autoencoder_encode(conditioning_image)

        # Create the concatenated conditioning tensor to be fed to `c_concat`
        conditioning_mask = torch.nn.functional.interpolate(conditioning_mask, size=latent_image.shape[-2:])
        conditioning_mask = conditioning_mask.expand(conditioning_image.shape[0], -1, -1, -1)
        image_conditioning = torch.cat([conditioning_mask, conditioning_image], dim=1)
        # image_conditioning = image_conditioning.to(shared.device).type(self.sd_model.dtype)
        image_conditioning = image_conditioning.to(DEVICE).type(torch.float32)

        return image_conditioning

    def img2img_image_conditioning(self, source_image, latent_image, image_mask=None):
        return latent_image.new_zeros(latent_image.shape[0], 5, 1, 1)
        # source_image = devices.cond_cast_float(source_image)

        # # HACK: Using introspection as the Depth2Image model doesn't appear to uniquely
        # # identify itself with a field common to all models. The conditioning_key is also hybrid.
        # if isinstance(self.sd_model, LatentDepth2ImageDiffusion):
        #     return self.depth2img_image_conditioning(source_image)

        # if self.sd_model.cond_stage_key == "edit":
        #     return self.edit_image_conditioning(source_image)

        # if self.sampler.conditioning_key in {'hybrid', 'concat'}:
        #     return self.inpainting_image_conditioning(source_image, latent_image, image_mask=image_mask)

        # if self.sampler.conditioning_key == "crossattn-adm":
        #     return self.unclip_image_conditioning(source_image)

        # # Dummy zero conditioning if we're not using inpainting or depth model.
        # return latent_image.new_zeros(latent_image.shape[0], 5, 1, 1)

    def init(self, all_prompts, all_seeds, all_subseeds):
        pass

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        raise NotImplementedError()

    def close(self):
        self.sampler = None
        self.c = None
        self.uc = None
        # if not opts.persistent_cond_cache:
        #     StableDiffusionProcessing.cached_c = [None, None]
        #     StableDiffusionProcessing.cached_uc = [None, None]
        StableDiffusionProcessing.cached_c = [None, None]
        StableDiffusionProcessing.cached_uc = [None, None]

    # def get_token_merging_ratio(self, for_hr=False):
    #     if for_hr:
    #         return self.token_merging_ratio_hr or opts.token_merging_ratio_hr or self.token_merging_ratio or opts.token_merging_ratio

    #     return self.token_merging_ratio or opts.token_merging_ratio

    def setup_prompts(self):
        if isinstance(self.prompt,list):
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
            raise RuntimeError(f"Received a different number of prompts ({len(self.all_prompts)}) and negative prompts ({len(self.all_negative_prompts)})")

        # self.all_prompts = [shared.prompt_styles.apply_styles_to_prompt(x, self.styles) for x in self.all_prompts]
        # self.all_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(x, self.styles) for x in self.all_negative_prompts]

        self.main_prompt = self.all_prompts[0]
        self.main_negative_prompt = self.all_negative_prompts[0]

    # def cached_params(self, required_prompts, steps, extra_network_data, hires_steps=None, use_old_scheduling=False):
    #     """Returns parameters that invalidate the cond cache if changed"""

    #     return (
    #         required_prompts,
    #         steps,
    #         hires_steps,
    #         use_old_scheduling,
    #         opts.CLIP_stop_at_last_layers,
    #         # shared.sd_model.sd_checkpoint_info,
    #         SD_MODEL.sd_checkpoint_info,
    #         extra_network_data,
    #         opts.sdxl_crop_left,
    #         opts.sdxl_crop_top,
    #         self.width,
    #         self.height,
    #     )

    # def get_conds_with_caching(self, function, required_prompts, steps, caches, extra_network_data, hires_steps=None):
    #     """
    #     Returns the result of calling function(shared.sd_model, required_prompts, steps)
    #     using a cache to store the result if the same arguments have been used before.

    #     cache is an array containing two elements. The first element is a tuple
    #     representing the previously used arguments, or None if no arguments
    #     have been used before. The second element is where the previously
    #     computed result is stored.

    #     caches is a list with items described above.
    #     """

    #     # if shared.opts.use_old_scheduling:
    #     # if opts.use_old_scheduling:
    #     #     old_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(required_prompts, steps, hires_steps, False)
    #     #     new_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(required_prompts, steps, hires_steps, True)
    #     #     if old_schedules != new_schedules:
    #     #         self.extra_generation_params["Old prompt editing timelines"] = True

    #     # cached_params = self.cached_params(required_prompts, steps, extra_network_data, hires_steps, shared.opts.use_old_scheduling)
    #     cached_params = self.cached_params(required_prompts, steps, extra_network_data, hires_steps, opts.use_old_scheduling)

    #     for cache in caches:
    #         if cache[0] is not None and cached_params == cache[0]:
    #             return cache[1]

    #     cache = caches[0]

    #     with devices.autocast():
    #         # cache[1] = function(shared.sd_model, required_prompts, steps, hires_steps, shared.opts.use_old_scheduling)
    #         cache[1] = function(SD_MODEL, required_prompts, steps, hires_steps, opts.use_old_scheduling)

    #     cache[0] = cached_params
    #     return cache[1]

    # def setup_conds(self):
    #     prompts = prompt_parser.SdConditioning(self.prompts, width=self.width, height=self.height)
    #     negative_prompts = prompt_parser.SdConditioning(self.negative_prompts, width=self.width, height=self.height, is_negative_prompt=True)

    #     sampler_config = sd_samplers.find_sampler_config(self.sampler_name)
    #     total_steps = sampler_config.total_steps(self.steps) if sampler_config else self.steps
    #     self.step_multiplier = total_steps // self.steps
    #     self.firstpass_steps = total_steps

        # self.uc = self.get_conds_with_caching(prompt_parser.get_learned_conditioning, negative_prompts, total_steps, [self.cached_uc], self.extra_network_data)
        # self.c = self.get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, prompts, total_steps, [self.cached_c], self.extra_network_data)

        embedded_prompts = self.prompt_embedding_vectors(prompt_array=self.all_prompts)
        embedded_prompts_cpu = embedded_prompts.to("cpu")
        embedded_prompts_list = embedded_prompts_cpu.detach().numpy()

        prompt_embedding = torch.tensor(embedded_prompts_list[0], dtype=torch.float32)
        prompt_embedding = prompt_embedding.view(1, 77, 768).to(DEVICE)

        self.uc = self.prompt_embedding_vectors([""])[0]
        self.c = prompt_embedding

    # def get_conds(self):
    #     return self.c, self.uc

    # def parse_extra_network_prompts(self):
    #     self.prompts, self.extra_network_data = extra_networks.parse_prompts(self.prompts)

    # def save_samples(self) -> bool:
    #     """Returns whether generated images need to be written to disk"""
    #     return opts.samples_save and not self.do_not_save_samples and (opts.save_incomplete_images or not state.interrupted and not state.skipped)

class Processed:
    def __init__(self, p: StableDiffusionProcessing, images_list, seed=-1, info="", subseed=None, all_prompts=None, all_negative_prompts=None, all_seeds=None, all_subseeds=None, index_of_first_image=0, infotexts=None, comments=""):
        self.images = images_list
        self.prompt = p.prompt
        self.negative_prompt = p.negative_prompt
        self.seed = seed
        self.subseed = subseed
        self.subseed_strength = p.subseed_strength
        self.info = info
        self.comments = "".join(f"{comment}\n" for comment in p.comments)
        self.width = p.width
        self.height = p.height
        self.sampler_name = p.sampler_name
        self.cfg_scale = p.cfg_scale
        self.image_cfg_scale = getattr(p, 'image_cfg_scale', None)
        self.steps = p.steps
        self.batch_size = p.batch_size
        self.restore_faces = p.restore_faces
        self.face_restoration_model = opts.face_restoration_model if p.restore_faces else None
        self.sd_model_name = p.sd_model_name
        self.sd_model_hash = p.sd_model_hash
        self.sd_vae_name = p.sd_vae_name
        self.sd_vae_hash = p.sd_vae_hash
        self.seed_resize_from_w = p.seed_resize_from_w
        self.seed_resize_from_h = p.seed_resize_from_h
        self.denoising_strength = getattr(p, 'denoising_strength', None)
        self.extra_generation_params = p.extra_generation_params
        self.index_of_first_image = index_of_first_image
        self.styles = p.styles
        # self.job_timestamp = state.job_timestamp
        self.clip_skip = opts.CLIP_stop_at_last_layers
        self.token_merging_ratio = p.token_merging_ratio
        self.token_merging_ratio_hr = p.token_merging_ratio_hr

        self.eta = p.eta
        self.ddim_discretize = p.ddim_discretize
        # self.s_churn = p.s_churn
        # self.s_tmin = p.s_tmin
        # self.s_tmax = p.s_tmax
        # self.s_noise = p.s_noise
        # self.s_min_uncond = p.s_min_uncond
        self.sampler_noise_scheduler_override = p.sampler_noise_scheduler_override
        self.prompt = self.prompt if not isinstance(self.prompt, list) else self.prompt[0]
        self.negative_prompt = self.negative_prompt if not isinstance(self.negative_prompt, list) else self.negative_prompt[0]
        self.seed = int(self.seed if not isinstance(self.seed, list) else self.seed[0]) if self.seed is not None else -1
        self.subseed = int(self.subseed if not isinstance(self.subseed, list) else self.subseed[0]) if self.subseed is not None else -1
        self.is_using_inpainting_conditioning = p.is_using_inpainting_conditioning

        self.all_prompts = all_prompts or p.all_prompts or [self.prompt]
        self.all_negative_prompts = all_negative_prompts or p.all_negative_prompts or [self.negative_prompt]
        self.all_seeds = all_seeds or p.all_seeds or [self.seed]
        self.all_subseeds = all_subseeds or p.all_subseeds or [self.subseed]
        self.infotexts = infotexts or [info]

    def js(self):
        obj = {
            "prompt": self.all_prompts[0],
            "all_prompts": self.all_prompts,
            "negative_prompt": self.all_negative_prompts[0],
            "all_negative_prompts": self.all_negative_prompts,
            "seed": self.seed,
            "all_seeds": self.all_seeds,
            "subseed": self.subseed,
            "all_subseeds": self.all_subseeds,
            "subseed_strength": self.subseed_strength,
            "width": self.width,
            "height": self.height,
            "sampler_name": self.sampler_name,
            "cfg_scale": self.cfg_scale,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "restore_faces": self.restore_faces,
            "face_restoration_model": self.face_restoration_model,
            "sd_model_name": self.sd_model_name,
            "sd_model_hash": self.sd_model_hash,
            "sd_vae_name": self.sd_vae_name,
            "sd_vae_hash": self.sd_vae_hash,
            "seed_resize_from_w": self.seed_resize_from_w,
            "seed_resize_from_h": self.seed_resize_from_h,
            "denoising_strength": self.denoising_strength,
            "extra_generation_params": self.extra_generation_params,
            "index_of_first_image": self.index_of_first_image,
            "infotexts": self.infotexts,
            "styles": self.styles,
            # "job_timestamp": self.job_timestamp,
            "clip_skip": self.clip_skip,
            "is_using_inpainting_conditioning": self.is_using_inpainting_conditioning,
        }

        return json.dumps(obj)

    def infotext(self, p: StableDiffusionProcessing, index):
        return create_infotext(p, self.all_prompts, self.all_seeds, self.all_subseeds, comments=[], position_in_batch=index % self.batch_size, iteration=index // self.batch_size)

    def get_token_merging_ratio(self, for_hr=False):
        return self.token_merging_ratio_hr if for_hr else self.token_merging_ratio

@dataclass(repr=False)
class StableDiffusionProcessingImg2Img(StableDiffusionProcessing):
    init_images: list = None
    # resize_mode: int = 0
    # denoising_strength: float = 0.75
    image_cfg_scale: float = None
    mask: Any = None
    mask_blur_x: int = 4
    mask_blur_y: int = 4
    mask_blur: int = None
    inpainting_fill: int = 0
    # inpaint_full_res: bool = True
    # inpaint_full_res_padding: int = 0
    # inpainting_mask_invert: int = 0
    # initial_noise_multiplier: float = None
    latent_mask: Image = None

    # image_mask: Any = field(default=None, init=False)

    # nmask: torch.Tensor = field(default=None, init=False)
    # image_conditioning: torch.Tensor = field(default=None, init=False)
    # init_img_hash: str = field(default=None, init=False)
    # mask_for_overlay: Image = field(default=None, init=False)
    # init_latent: torch.Tensor = field(default=None, init=False)

    def __post_init__(self):
        super().__post_init__()

        self.image_mask = self.mask
        self.mask = None
        # self.initial_noise_multiplier = opts.initial_noise_multiplier if self.initial_noise_multiplier is None else self.initial_noise_multiplier

    # NOTE: Check if bug
    # NOTE: Doesn't run if we comment this.
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
        # self.image_cfg_scale: float = self.image_cfg_scale if shared.sd_model.cond_stage_key == "edit" else None
        # self.image_cfg_scale: float = self.image_cfg_scale if SD_MODEL.cond_stage_key == "edit" else None
        self.image_cfg_scale: float = self.image_cfg_scale

        # self.sampler = sd_samplers.create_sampler(self.sampler_name, self.sd_model)
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

            # if self.inpainting_mask_invert:
            #     image_mask = ImageOps.invert(image_mask)

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

            # if self.inpaint_full_res:
            #     self.mask_for_overlay = image_mask
            #     mask = image_mask.convert('L')
            #     crop_region = masking.get_crop_region(np.array(mask), self.inpaint_full_res_padding)
            #     crop_region = masking.expand_crop_region(crop_region, self.width, self.height, mask.width, mask.height)
            #     x1, y1, x2, y2 = crop_region

            #     mask = mask.crop(crop_region)
            #     image_mask = images.resize_image(2, mask, self.width, self.height)
            #     self.paste_to = (x1, y1, x2-x1, y2-y1)
            # else:
            # image_mask = images.resize_image(self.resize_mode, image_mask, self.width, self.height)
            np_mask = np.array(image_mask)
            np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.uint8)
            self.mask_for_overlay = Image.fromarray(np_mask)

            self.overlay_images = []

        latent_mask = self.latent_mask if self.latent_mask is not None else image_mask

        add_color_corrections = opts.img2img_color_correction and self.color_corrections is None
        if add_color_corrections:
            self.color_corrections = []
        imgs = []
        print(self.init_images)
        for img in self.init_images:

            # Save init image
            if opts.save_init_img:
                self.init_img_hash = hashlib.md5(img.tobytes()).hexdigest()
                images.save_image(img, path=opts.outdir_init_images, basename=None, forced_filename=self.init_img_hash, save_to_dirs=False)

            # image = images.flatten(img, opts.img2img_background_color)
            image = flatten(img, opts.img2img_background_color)

            # if crop_region is None and self.resize_mode != 3:
            #     image = images.resize_image(self.resize_mode, image, self.width, self.height)

            if image_mask is not None:
                image_masked = Image.new('RGBa', (image.width, image.height))
                image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(self.mask_for_overlay.convert('L')))

                self.overlay_images.append(image_masked.convert('RGBA'))

            # crop_region is not None if we are doing inpaint full res
            if crop_region is not None:
                image = image.crop(crop_region)
                # image = images.resize_image(2, image, self.width, self.height)

            # if image_mask is not None:
            #     if self.inpainting_fill != 1:
            #         image = masking.fill(image, latent_mask)

            # if add_color_corrections:
            #     self.color_corrections.append(setup_color_correction(image))

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
        # image = image.to(shared.device, dtype=devices.dtype_vae)
        image = image.to(self.device, dtype=torch.float32)

        # if opts.sd_vae_encode_method != 'Full':
        #     self.extra_generation_params['VAE Encoder'] = opts.sd_vae_encode_method

        # self.init_latent = images_tensor_to_samples(image, approximation_indexes.get(opts.sd_vae_encode_method), self.sd_model)
        self.init_latent = self.model.autoencoder_encode(image)
        # devices.torch_gc()
        # NOTE: Assuming only cuda for now
        with torch.cuda.device('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # if self.resize_mode == 3:
        #     self.init_latent = torch.nn.functional.interpolate(self.init_latent, size=(self.height // opt_f, self.width // opt_f), mode="bilinear")

        if image_mask is not None:
            init_mask = latent_mask
            latmask = init_mask.convert('RGB').resize((self.init_latent.shape[3], self.init_latent.shape[2]))
            latmask = np.moveaxis(np.array(latmask, dtype=np.float32), 2, 0) / 255
            latmask = latmask[0]
            latmask = np.around(latmask)
            latmask = np.tile(latmask[None], (4, 1, 1))

            # self.mask = torch.asarray(1.0 - latmask).to(shared.device).type(self.sd_model.dtype)
            # self.nmask = torch.asarray(latmask).to(shared.device).type(self.sd_model.dtype)
            self.mask = torch.asarray(1.0 - latmask).to(DEVICE).type(self.sd_model.dtype)
            self.nmask = torch.asarray(latmask).to(DEVICE).type(self.sd_model.dtype)

            # this needs to be fixed to be done in sample() using actual seeds for batches
            if self.inpainting_fill == 2:
                self.init_latent = self.init_latent * self.mask + create_random_tensors(self.init_latent.shape[1:], all_seeds[0:self.init_latent.shape[0]]) * self.nmask
            elif self.inpainting_fill == 3:
                self.init_latent = self.init_latent * self.mask

        # self.image_conditioning = self.init_latent.new_zeros(self.init_latent.shape[0], 5, 1, 1)
        self.image_conditioning = self.inpainting_image_conditioning(image * 2 - 1, self.init_latent, image_mask)
        # self.image_conditioning = self.img2img_image_conditioning(image * 2 - 1, self.init_latent, image_mask)

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        # x = self.rng.next()
        # NOTE: This will crash, but problem for later
        x = create_random_tensors(shape=(1, 4, 64, 64))

        # if self.initial_noise_multiplier != 1.0:
        #     self.extra_generation_params["Noise multiplier"] = self.initial_noise_multiplier
        #     x *= self.initial_noise_multiplier

        # samples = self.sampler.sample_img2img(self, self.init_latent, x, conditioning, unconditional_conditioning, image_conditioning=self.image_conditioning)
        # pil_image = to_pil(x[0])
        # pil_image.save('output/inpainting/x.png')
        # init_latent = self.init_latent[:, :3, :, :]
        # pil_image = to_pil(init_latent[0])
        # pil_image.save('output/inpainting/orig_latent.png')
        # pil_image = to_pil(self.mask)
        # pil_image.save('output/inpainting/mask.png')
        # pil_image = to_pil(self.image_conditioning[:, :3, :, :][0])
        # pil_image.save('output/inpainting/image_conditioning.png')
        # pil_image = to_pil(conditioning)
        # pil_image.save('output/inpainting/conditioning.png')
        # pil_image = to_pil(unconditional_conditioning)
        # pil_image.save('output/inpainting/unconditional_conditioning.png')

        mask = torch.zeros_like(x, device=self.device)
        mask[:, :, mask.shape[2] // 2:, :] = 1.

        print('init_latent.shape', self.init_latent.shape)
        print('conditioning.shape', conditioning.shape)
        print('unconditional_conditioning.shape', unconditional_conditioning.shape)
        print('image_conditioning.shape', self.image_conditioning.shape)
        print('mask', mask.shape)
        print('x', x.shape)

        orig_noise = torch.randn(self.init_latent.shape, device=DEVICE)

        # samples = self.sampler.sample(shape=[1, 4, 64, 64],
        #                               cond=conditioning,
        #                               uncond_cond=unconditional_conditioning
        #                              )
        samples = self.sampler.paint(x=x,
                                     orig=self.init_latent,
                                     t_start=35,
                                     cond=conditioning,
                                     orig_noise=orig_noise,
                                     uncond_scale=0.1,
                                     # cond=self.image_conditioning,
                                     uncond_cond=unconditional_conditioning,
                                     mask=mask
                                     )

        images = self.sd.get_image_from_latent(samples)
        pil_image = to_pil(images[0])
        pil_image.save('output/inpainting/result_notlatent.png')

        # if self.mask is not None:
        #     samples = samples * self.nmask + self.init_latent * self.mask

        # del x
        # devices.torch_gc()
        with torch.cuda.device('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        return samples

    def get_token_merging_ratio(self, for_hr=False):
        return self.token_merging_ratio or ("token_merging_ratio" in self.override_settings and opts.token_merging_ratio) or opts.token_merging_ratio_img2img or opts.token_merging_ratio


def process_images_inner(p: StableDiffusionProcessing) -> Processed:
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""

    if isinstance(p.prompt, list):
        assert(len(p.prompt) > 0)
    else:
        assert p.prompt is not None

    # devices.torch_gc()
    # NOTE: Assuming only cuda for now
    with torch.cuda.device('cuda'):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # NOTE: Hardcoding them for now
    # seed = get_fixed_seed(p.seed)
    # subseed = get_fixed_seed(p.subseed)
    seed = 123
    subseed = 456

    # if p.restore_faces is None:
    #     p.restore_faces = opts.face_restoration

    # if p.tiling is None:
    #     p.tiling = opts.tiling

    # if p.refiner_checkpoint not in (None, "", "None", "none"):
    #     p.refiner_checkpoint_info = sd_models.get_closet_checkpoint_match(p.refiner_checkpoint)
    #     if p.refiner_checkpoint_info is None:
    #         raise Exception(f'Could not find checkpoint with name {p.refiner_checkpoint}')

    # p.sd_model_name = shared.sd_model.sd_checkpoint_info.name_for_extra
    # p.sd_model_hash = shared.sd_model.sd_model_hash
    # p.sd_vae_name = sd_vae.get_loaded_vae_name()
    # p.sd_vae_hash = sd_vae.get_loaded_vae_hash()

    # modules.sd_hijack.model_hijack.apply_circular(p.tiling)
    # modules.sd_hijack.model_hijack.clear_comments()

    p.setup_prompts()

    if isinstance(seed, list):
        p.all_seeds = seed
    else:
        p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]
        # p.all_seeds = [seed]

    if isinstance(subseed, list):
        p.all_subseeds = subseed
    else:
        p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]
        # p.all_subseeds = [subseed]

    # if os.path.exists(cmd_opts.embeddings_dir) and not p.do_not_reload_embeddings:
    #     model_hijack.embedding_db.load_textual_inversion_embeddings()

    # if p.scripts is not None:
    #     p.scripts.process(p)

    infotexts = []
    output_images = []

    # with torch.no_grad(), p.sd_model.ema_scope():
    with torch.no_grad():
        # with devices.autocast():
        with torch.autocast("cuda"):
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)

            # # for OSX, loading the model during sampling changes the generated picture, so it is loaded here
            # if shared.opts.live_previews_enable and opts.show_progress_type == "Approx NN":
            #     sd_vae_approx.model()

            # sd_unet.apply_unet()

        # if state.job_count == -1:
        #     state.job_count = p.n_iter

        for n in range(p.n_iter):
            p.iteration = n

            # if state.skipped:
            #     state.skipped = False

            # if state.interrupted:
            #     break

            # sd_models.reload_model_weights()  # model can be changed for example by refiner

            p.prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            p.negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            p.seeds = p.all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
            p.subseeds = p.all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

            # NOTE: Maybe needed, unless we can just use random tensors with the correct shape
            # p.rng = rng.ImageRNG((opt_C, p.height // opt_f, p.width // opt_f), p.seeds, subseeds=p.subseeds, subseed_strength=p.subseed_strength, seed_resize_from_h=p.seed_resize_from_h, seed_resize_from_w=p.seed_resize_from_w)

            # if p.scripts is not None:
            #     p.scripts.before_process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)

            if len(p.prompts) == 0:
                break

            # p.parse_extra_network_prompts()

            # if not p.disable_extra_networks:
            #     with devices.autocast():
            #         extra_networks.activate(p, p.extra_network_data)

            # if p.scripts is not None:
            #     p.scripts.process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)

            # params.txt should be saved after scripts.process_batch, since the
            # infotext could be modified by that callback
            # Example: a wildcard processed by process_batch sets an extra model
            # strength, which is saved as "Model Strength: 1.0" in the infotext
            # NOTE: Maybe needed, not sure what this does
            # if n == 0:
            #     with open(os.path.join(paths.data_path, "params.txt"), "w", encoding="utf8") as file:
            #         processed = Processed(p, [])
            #         file.write(processed.infotext(p, 0))

            # p.setup_conds()

            # for comment in model_hijack.comments:
            #     p.comment(comment)

            # p.extra_generation_params.update(model_hijack.extra_generation_params)

            # if p.n_iter > 1:
            #     shared.state.job = f"Batch {n+1} out of {p.n_iter}"

            # with devices.without_autocast() if devices.unet_needs_upcast else devices.autocast():
            samples_ddim = p.sample(conditioning=p.c, unconditional_conditioning=p.uc, seeds=p.seeds, subseeds=p.subseeds, subseed_strength=p.subseed_strength, prompts=p.prompts)

            # if getattr(samples_ddim, 'already_decoded', False):
            #     x_samples_ddim = samples_ddim
            # else:
            #     if opts.sd_vae_decode_method != 'Full':
            #         p.extra_generation_params['VAE Decoder'] = opts.sd_vae_decode_method

            #     x_samples_ddim = decode_latent_batch(p.sd_model, samples_ddim, target_device=devices.cpu, check_for_nans=True)
            x_samples_ddim = samples_ddim
            print('shape', x_samples_ddim.shape)
            pil_image = to_pil(x_samples_ddim[0])
            pil_image.save('output/inpainting/result.png')

            x_samples_ddim = torch.stack(x_samples_ddim).float()
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            del samples_ddim

            # if lowvram.is_enabled(shared.sd_model):
            #     lowvram.send_everything_to_cpu()

            # devices.torch_gc()
            with torch.cuda.device('cuda'):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            if p.scripts is not None:
                p.scripts.postprocess_batch(p, x_samples_ddim, batch_number=n)

                p.prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
                p.negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]

                batch_params = scripts.PostprocessBatchListArgs(list(x_samples_ddim))
                p.scripts.postprocess_batch_list(p, batch_params, batch_number=n)
                x_samples_ddim = batch_params.images

            def infotext(index=0, use_main_prompt=False):
                return create_infotext(p, p.prompts, p.seeds, p.subseeds, use_main_prompt=use_main_prompt, index=index, all_negative_prompts=p.negative_prompts)

            save_samples = p.save_samples()

            for i, x_sample in enumerate(x_samples_ddim):
                p.batch_index = i

                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)

                if p.restore_faces:
                    if save_samples and opts.save_images_before_face_restoration:
                        images.save_image(Image.fromarray(x_sample), p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(i), p=p, suffix="-before-face-restoration")

                    # devices.torch_gc()
                    with torch.cuda.device('cuda'):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

                    x_sample = modules.face_restoration.restore_faces(x_sample)
                    # devices.torch_gc()
                    with torch.cuda.device('cuda'):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

                image = Image.fromarray(x_sample)

                if p.scripts is not None:
                    pp = scripts.PostprocessImageArgs(image)
                    p.scripts.postprocess_image(p, pp)
                    image = pp.image
                if p.color_corrections is not None and i < len(p.color_corrections):
                    if save_samples and opts.save_images_before_color_correction:
                        image_without_cc = apply_overlay(image, p.paste_to, i, p.overlay_images)
                        images.save_image(image_without_cc, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(i), p=p, suffix="-before-color-correction")
                    image = apply_color_correction(p.color_corrections[i], image)

                image = apply_overlay(image, p.paste_to, i, p.overlay_images)

                if save_samples:
                    images.save_image(image, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(i), p=p)

                text = infotext(i)
                infotexts.append(text)
                if opts.enable_pnginfo:
                    image.info["parameters"] = text
                output_images.append(image)
                if save_samples and hasattr(p, 'mask_for_overlay') and p.mask_for_overlay and any([opts.save_mask, opts.save_mask_composite, opts.return_mask, opts.return_mask_composite]):
                    image_mask = p.mask_for_overlay.convert('RGB')
                    image_mask_composite = Image.composite(image.convert('RGBA').convert('RGBa'), Image.new('RGBa', image.size), images.resize_image(2, p.mask_for_overlay, image.width, image.height).convert('L')).convert('RGBA')

                    if opts.save_mask:
                        images.save_image(image_mask, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(i), p=p, suffix="-mask")

                    if opts.save_mask_composite:
                        images.save_image(image_mask_composite, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(i), p=p, suffix="-mask-composite")

                    if opts.return_mask:
                        output_images.append(image_mask)

                    if opts.return_mask_composite:
                        output_images.append(image_mask_composite)

            del x_samples_ddim

            # devices.torch_gc()
            with torch.cuda.device('cuda'):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            # state.nextjob()

        p.color_corrections = None

        index_of_first_image = 0
        unwanted_grid_because_of_img_count = len(output_images) < 2 and opts.grid_only_if_multiple
        if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
            grid = images.image_grid(output_images, p.batch_size)

            if opts.return_grid:
                text = infotext(use_main_prompt=True)
                infotexts.insert(0, text)
                if opts.enable_pnginfo:
                    grid.info["parameters"] = text
                output_images.insert(0, grid)
                index_of_first_image = 1
            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", p.all_seeds[0], p.all_prompts[0], opts.grid_format, info=infotext(use_main_prompt=True), short_filename=not opts.grid_extended_filename, p=p, grid=True)

    if not p.disable_extra_networks and p.extra_network_data:
        extra_networks.deactivate(p, p.extra_network_data)

    # devices.torch_gc()
    with torch.cuda.device('cuda'):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    res = Processed(
        p,
        images_list=output_images,
        seed=p.all_seeds[0],
        info=infotexts[0],
        subseed=p.all_subseeds[0],
        index_of_first_image=index_of_first_image,
        infotexts=infotexts,
    )

    if p.scripts is not None:
        p.scripts.postprocess(p, res)

    return res

def process_images(p: StableDiffusionProcessing) -> Processed:
    # if p.scripts is not None:
    #     p.scripts.before_process(p)

    # stored_opts = {k: opts.data[k] for k in p.override_settings.keys()}

    try:
        # if no checkpoint override or the override checkpoint can't be found, remove override entry and load opts checkpoint
        # and if after running refiner, the refiner model is not unloaded - webui swaps back to main model here, if model over is present it will be reloaded afterwards
        # if sd_models.checkpoint_aliases.get(p.override_settings.get('sd_model_checkpoint')) is None:
        #     p.override_settings.pop('sd_model_checkpoint', None)
        #     sd_models.reload_model_weights()

        # for k, v in p.override_settings.items():
        #     opts.set(k, v, is_api=True, run_callbacks=False)

        #     if k == 'sd_model_checkpoint':
        #         sd_models.reload_model_weights()

        #     if k == 'sd_vae':
        #         sd_vae.reload_vae_weights()

        # sd_models.apply_token_merging(p.sd_model, p.get_token_merging_ratio())

        res = process_images_inner(p)

    finally:
        # NOTE: Added just to not deal with indentation after removing try/finally
        print('')
        # sd_models.apply_token_merging(p.sd_model, 0)

        # restore opts to original state
        # if p.override_settings_restore_afterwards:
        #     for k, v in stored_opts.items():
        #         setattr(opts, k, v)

        #         if k == 'sd_vae':
        #             sd_vae.reload_vae_weights()

    return res

# def img2img(id_task: str, mode: int, prompt: str, negative_prompt: str, prompt_styles, init_img, sketch, init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig, init_img_inpaint, init_mask_inpaint, steps: int, sampler_name: str, mask_blur: int, mask_alpha: float, inpainting_fill: int, n_iter: int, batch_size: int, cfg_scale: float, image_cfg_scale: float, denoising_strength: float, selected_scale_tab: int, height: int, width: int, scale_by: float, resize_mode: int, inpaint_full_res: bool, inpaint_full_res_padding: int, inpainting_mask_invert: int, img2img_batch_input_dir: str, img2img_batch_output_dir: str, img2img_batch_inpaint_mask_dir: str, override_settings_texts, img2img_batch_use_png_info: bool, img2img_batch_png_info_props: list, img2img_batch_png_info_dir: str, request: gr.Request, *args):
def img2img(prompt: str, negative_prompt: str, sampler_name: str, batch_size: int, n_iter: int, steps: int, cfg_scale: float, width: int, height: int, mask_blur: int, inpainting_fill: int, init_img
            #resize_mode: int,
            # denoising_strength: float,
            # image_cfg_scale: float,
            # inpaint_full_res: bool,
            # inpaint_full_res_padding: int,
            # inpainting_mask_invert: int
            ):
    # override_settings = create_override_settings_dict(override_settings_texts)

    # is_batch = mode == 5

    # if mode == 0:  # img2img
    #     image = init_img
    #     mask = None
    # elif mode == 1:  # img2img sketch
    #     image = sketch
    #     mask = None
    # elif mode == 2:  # inpaint
    #     image, mask = init_img_with_mask["image"], init_img_with_mask["mask"]
    #     mask = processing.create_binary_mask(mask)
    # elif mode == 3:  # inpaint sketch
    #     image = inpaint_color_sketch
    #     orig = inpaint_color_sketch_orig or inpaint_color_sketch
    #     pred = np.any(np.array(image) != np.array(orig), axis=-1)
    #     mask = Image.fromarray(pred.astype(np.uint8) * 255, "L")
    #     mask = ImageEnhance.Brightness(mask).enhance(1 - mask_alpha / 100)
    #     blur = ImageFilter.GaussianBlur(mask_blur)
    #     image = Image.composite(image.filter(blur), orig, mask.filter(blur))
    # elif mode == 4:  # inpaint upload mask
    #     image = init_img_inpaint
    #     mask = init_mask_inpaint
    # else:
    #     image = None
    #     mask = None
    image = init_img
    mask = None

    # # Use the EXIF orientation of photos taken by smartphones.
    # if image is not None:
    #     image = ImageOps.exif_transpose(image)

    # if selected_scale_tab == 1 and not is_batch:
    #     assert image, "Can't scale by because no image is selected"

    #     width = int(image.width * scale_by)
    #     height = int(image.height * scale_by)

    # assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'

    p = StableDiffusionProcessingImg2Img(
        # sd_model=shared.sd_model,
        sd_model=SD_MODEL,
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        # outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        # styles=prompt_styles,
        sampler_name=sampler_name,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        init_images=[image],
        mask=mask,
        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        # resize_mode=resize_mode,
        # denoising_strength=denoising_strength,
        # image_cfg_scale=image_cfg_scale,
        # inpaint_full_res=inpaint_full_res,
        # inpaint_full_res_padding=inpaint_full_res_padding,
        # inpainting_mask_invert=inpainting_mask_invert,
        # override_settings=override_settings,
    )

    # p.scripts = modules.scripts.scripts_img2img
    # p.script_args = args

    # p.user = request.username

    # if shared.cmd_opts.enable_console_prompts:
    #     print(f"\nimg2img: {prompt}", file=shared.progress_print_out)

    if mask:
        p.extra_generation_params["Mask blur"] = mask_blur

    with closing(p):
        # if is_batch:
        #     assert not shared.cmd_opts.hide_ui_dir_config, "Launched with --hide-ui-dir-config, batch img2img disabled"

        #     process_batch(p, img2img_batch_input_dir, img2img_batch_output_dir, img2img_batch_inpaint_mask_dir, args, to_scale=selected_scale_tab == 1, scale_by=scale_by, use_png_info=img2img_batch_use_png_info, png_info_props=img2img_batch_png_info_props, png_info_dir=img2img_batch_png_info_dir)

        #     processed = Processed(p, [], p.seed, "")
        # else:
        # processed = modules.scripts.scripts_img2img.run(p, *args)
        # if processed is None:
        processed = process_images(p)

    # shared.total_tqdm.clear()
    # TOTAL_TQDM.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    # return processed.images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")
    return processed.images, generation_info_js, processed.info, processed.comments

image = Image.open("test.png")

img2img(prompt="Sample prompt",
        negative_prompt="Negative sample prompt",
        sampler_name="ddim",
        batch_size=1,
        n_iter=5,
        steps=5,
        cfg_scale=7.1,
        width=512,
        height=512,
        mask_blur=4,
        inpainting_fill=4,
        init_img=image
        # resize_mode=3,
        # denoising_strength=0.75,
        # image_cfg_scale=0.5,
        # inpaint_full_res=False,
        # inpaint_full_res_padding=10,
        # inpainting_mask_invert=5
        )
