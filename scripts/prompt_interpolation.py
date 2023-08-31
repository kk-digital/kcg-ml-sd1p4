import os
import sys

base_dir = os.getcwd()
sys.path.insert(0, base_dir)

import torch

from configs.model_config import ModelPathConfig
from stable_diffusion import StableDiffusion, SDconfigs
from stable_diffusion.utils_backend import get_device
from stable_diffusion.utils_image import to_pil
from ga.utils import get_next_ga_dir
import ga

N_STEPS = 20  # 20, 12
CFG_STRENGTH = 9

DEVICE = get_device()
config = ModelPathConfig()

# Load Stable Diffusion
sd = StableDiffusion(device=DEVICE, n_steps=N_STEPS)
sd.quick_initialize().load_autoencoder(config.get_model(SDconfigs.VAE)).load_decoder(config.get_model(SDconfigs.VAE_DECODER))
sd.model.load_unet(config.get_model(SDconfigs.UNET))

def prompt_embedding_vectors(sd, prompt_array):
    # Generate embeddings for each prompt
    embedded_prompts = ga.clip_text_get_prompt_embedding(config, prompts=prompt_array)
    embedded_prompts.to("cpu")
    return embedded_prompts

def linear_interpolation(A, B, t):
    interpolated_vector = []
    for i in range(len(A)):
        interpolated_dim = A[i] + t * (B[i] - A[i])
        interpolated_vector.append(interpolated_dim)
    return interpolated_vector


# Get embedding of null prompt
NULL_PROMPT = prompt_embedding_vectors(sd, [""])[0]

# generate prompts and get embeddings
prompt_phrase_length = 10
prompts_count = 2
prompts_array = ga.generate_prompts(prompts_count, prompt_phrase_length)

# get prompt_str array
prompts_str_array = []
for prompt in prompts_array:
    prompt_str = prompt.get_prompt_str()
    prompts_str_array.append(prompt_str)

embedded_prompts = prompt_embedding_vectors(sd, prompt_array=prompts_str_array)

embedded_prompts_cpu = embedded_prompts.to("cpu")
embedded_prompts_list = embedded_prompts_cpu.detach().numpy()

embedding1 = embedded_prompts_list[0]
embedding2 = embedded_prompts_list[1]

output_dir = os.path.join('output', 'prompt_interpolation')
os.makedirs(output_dir)

for t in range(0, 11, 0.1):
    t_decimal = t / 10.0
    interpolation = torch.tensor(linear_interpolation(embedding1, embedding2, t_decimal), dtype=torch.float32, device=DEVICE)
    latent = sd.generate_images_latent_from_embeddings(
        seed=123,
        embedded_prompt=interpolation,
        null_prompt=NULL_PROMPT,
        uncond_scale=CFG_STRENGTH
    )

    image = sd.get_image_from_latent(latent)

    # move to gpu and cleanup
    interpolation.to("cpu")
    del interpolation

    pil_image = to_pil(image[0])
    filename = os.path.join(output_dir, f'{t}.png')
    pil_image.save(filename)
