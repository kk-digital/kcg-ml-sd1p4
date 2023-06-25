import os
import argparse
import torch
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
from collections import defaultdict
from diffusers.loaders import LoraLoaderMixin
from diffusers import DPMSolverMultistepScheduler
from diffusers import DDIMScheduler
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import LMSDiscreteScheduler

class Txt2ImgLoRa:
    def __init__(self, prompt, output, checkpoint_path, lora_path, steps, scale, force_cpu, sampler):
        self.prompt = prompt
        self.output = output
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.lora_path = os.path.abspath(lora_path)
        self.steps = steps
        self.scale = scale
        self.force_cpu = force_cpu
        self.sampler = sampler

        self.pipe = None

    def set_scheduler(self):
        if self.sampler == "euler":
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        elif self.sampler == "dpm":
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        elif self.sampler == "lms":
            self.pipe.scheduler = LMSDiscreteScheduler.from_config(self.pipe.scheduler.config)
        elif self.sampler == "ddim":
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

    def initialize_pipeline(self):
        torch_dtype = torch.float32 if self.force_cpu else torch.float16
        self.pipe = StableDiffusionPipeline.from_ckpt(self.checkpoint_path, torch_dtype=torch_dtype,
                                                      safety_checker = None, requires_safety_checker = False)
        lora_path = os.path.abspath(self.lora_path)
        if not self.force_cpu:
            self.pipe.to("cuda")
            cuda = "cuda"
        else:
            cuda = "cpu"

        self.pipe.load_lora_weights(lora_path, lora_weight=0.6)
        #self.pipe = load_lora_weights(self.pipe, self.lora_path)
        self.pipe.safety_checker = None
        self.set_scheduler()

    def generate_image(self):
        image = self.pipe(self.prompt, num_inference_steps=self.steps,
                          guidance_scale=self.scale).images[0]
        image.save(self.output)

    def run(self):
        self.initialize_pipeline()
        self.generate_image()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an image from text using Stable Diffusion and LoRa.")
    parser.add_argument("--prompt", type=str, help="The text prompt describing the desired image.")
    parser.add_argument("--output", type=str, default="output.png", help="Output file path. Default: output.png")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the base Stable Diffusion checkpoint file.")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the LoRa checkpoint file.")
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps. Default: 30")
    parser.add_argument("--scale", type=float, default=7.5, help="Guidance scale. Default: 7.5")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage instead of GPU.")
    parser.add_argument("--sampler", type=str, choices = ['euler', 'dpm', 'lms', 'ddim'], default="dpm", help="Sampler to use when generating the image")

    args = parser.parse_args()

    txt2img = Txt2ImgLoRa(
        args.prompt,
        args.output,
        args.checkpoint_path,
        args.lora_path,
        args.steps,
        args.scale,
        args.force_cpu,
        args.sampler
    )
    txt2img.run()
