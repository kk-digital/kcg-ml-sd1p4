import os
import argparse
import torch
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
from collections import defaultdict
from diffusers.loaders import LoraLoaderMixin

def load_lora_weights(pipeline, checkpoint_path):
    # load base model
    pipeline.to("cuda")
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    alpha = 0.75
    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path, device="cuda")
    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)

    return pipeline

class Txt2ImgLoRa:
    def __init__(self, prompt, output, checkpoint_path, lora_path, steps, scale, force_cpu):
        self.prompt = prompt
        self.output = output
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.lora_path = os.path.abspath(lora_path)
        self.steps = steps
        self.scale = scale
        self.force_cpu = force_cpu

        self.pipe = None

    def initialize_pipeline(self):
        torch_dtype = torch.float32 if self.force_cpu else torch.float16
        self.pipe = StableDiffusionPipeline.from_ckpt(self.checkpoint_path, torch_dtype=torch_dtype,
                                                      safety_checker = None, requires_safety_checker = False)
        self.pipe = load_lora_weights(self.pipe, self.lora_path)
        self.pipe.safety_checker = None

        if not self.force_cpu:
            self.pipe.to("cuda")
            cuda = "cuda"
        else:
            cuda = "cpu"

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

    args = parser.parse_args()

    txt2img = Txt2ImgLoRa(
        args.prompt,
        args.output,
        args.checkpoint_path,
        args.lora_path,
        args.steps,
        args.scale,
        args.force_cpu
    )
    txt2img.run()
