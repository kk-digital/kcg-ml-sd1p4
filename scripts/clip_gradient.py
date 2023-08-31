import argparse
import os
import sys
import time
import clip
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

base_dir = os.getcwd()
sys.path.insert(0, base_dir)

from configs.model_config import ModelPathConfig
from stable_diffusion import StableDiffusion
from stable_diffusion.utils_backend import get_device
from stable_diffusion.utils_image import to_pil
from stable_diffusion.model_paths import (SDconfigs)

import ga
from ga.fitness_chad_score import compute_chad_score_from_pil
from ga.model_clip_text import clip_text_get_prompt_embedding

def parse_arguments():
    parser = argparse.ArgumentParser(description="Use gradients to optimize clip vector.")

    parser.add_argument('--iterations', type=int,
                        default=1000,
                        help='How many iterations to perform')
    parser.add_argument('--learning_rate', type=float,
                        default=0.0001,
                        help='Learning rate to use when optimizing')

    return parser.parse_args()


args = parse_arguments()

config = ModelPathConfig()

N_STEPS = 10
DEVICE = get_device()

sd = StableDiffusion(device=DEVICE, n_steps=N_STEPS)
sd.quick_initialize().load_autoencoder(config.get_model(SDconfigs.VAE)).load_decoder(config.get_model(SDconfigs.VAE_DECODER))
sd.model.load_unet(config.get_model(SDconfigs.UNET))
sd.initialize_latent_diffusion(path='input/model/sd/v1-5-pruned-emaonly/v1-5-pruned-emaonly.safetensors', force_submodels_init=True)

CFG_STRENGTH = 9

# load clip
image_features_clip_model, preprocess = clip.load("ViT-L/14", device=DEVICE)

def random_tensor(shape=(1, 4, 64, 64), low=0.0, high=1.0, device=DEVICE, requires_grad=False):
    random_tensor = torch.tensor(np.random.uniform(low=low, high=high, size=shape), dtype=torch.float32, device=device, requires_grad=requires_grad)
    return random_tensor

def prompt_embedding_vectors(sd, prompt_array):
    embedded_prompts = ga.clip_text_get_prompt_embedding(config, prompts=prompt_array)
    embedded_prompts.to("cpu")
    return embedded_prompts

def get_image_features(
        image, model, preprocess, device=DEVICE,
):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    image_features_cpu = image_features.cpu()

    # free gpu memory
    image_features = image_features.detach()
    del image_features
    torch.cuda.empty_cache()

    return image_features_cpu.numpy()

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, input1, input2):
        input1_norm = F.normalize(input1, dim=-1)
        input2_norm = F.normalize(input2, dim=-1)

        # Compute the cosine similarity
        cos_sim = nn.functional.cosine_similarity(input1_norm, input2_norm, dim=-1)
        loss = -cos_sim.mean()
        # cosine_similarity = torch.sum(input1_norm * input2_norm, dim=-1)
        # loss = cosine_similarity.mean()

        return loss

text = clip.tokenize(["a dog"]).to(DEVICE)
target_clip_text = image_features_clip_model.encode_text(text).shape


iterations = args.iterations
learning_rate = args.learning_rate

prompts_array = ga.generate_prompts(1, 10)

# get prompt_str array
prompts_str_array = []
for prompt in prompts_array:
    prompt_str = prompt.get_positive_prompt_str()
    prompts_str_array.append(prompt_str)

embedded_prompts = prompt_embedding_vectors(sd, prompt_array=prompts_str_array)
embedded_prompts_cpu = embedded_prompts.to("cpu")
embedded_prompts_list = embedded_prompts_cpu.detach().numpy()

prompt_embedding = torch.tensor(embedded_prompts_list[0], dtype=torch.float32)
prompt_embedding = prompt_embedding.view(1, 77, 768).to(DEVICE)

NULL_PROMPT = prompt_embedding_vectors(sd, [""])[0]

target_latent = sd.generate_images_latent_from_embeddings(
        seed=123,
        embedded_prompt=prompt_embedding,
        null_prompt=NULL_PROMPT,
        uncond_scale=CFG_STRENGTH
    )

# random_clip = random_tensor(shape=(1, 768), low=-1.0, high=1.0, requires_grad=True)
random_latent = random_tensor(shape=(1, 4, 64, 64), low=-1.0, high=1.0, requires_grad=True)
# optimizer = optim.Adam([latent], lr=learning_rate)
optimizer = optim.SGD([random_latent], lr=learning_rate)
# loss = nn.MSELoss(reduction='sum')
cosine_loss = CosineSimilarityLoss()

# with torch.no_grad():
image_target = sd.model.autoencoder_decode(target_latent)
pil_image_target = to_pil(image_target[0])
target_clip_image = get_image_features(pil_image_target, image_features_clip_model, preprocess)

target_clip_image = torch.tensor(target_clip_image, dtype=torch.float32, device=DEVICE)

target_latent = target_latent.detach()
image_target = sd.model.autoencoder_decode(target_latent)
image_target = image_target.detach()
pil_image_target = to_pil(image_target[0])
clip_target = get_image_features(pil_image_target, image_features_clip_model, preprocess).squeeze()
clip_target = torch.tensor(clip_target, dtype=torch.float32, requires_grad=True, device=DEVICE).detach()

start_time = time.time()
for i in range(0, iterations):
    # loss = loss(random_clip, target_clip_image)
    decode_start_time = time.time()
    image_output = sd.model.autoencoder_decode(random_latent)
    decode_end_time = time.time()

    elapsed_time = decode_end_time - decode_start_time
    print(f"Time elapsed (decode): {elapsed_time:.4f} seconds")

    pil_image_output = to_pil(image_output[0])
    clip_start_time = time.time()
    clip_output = get_image_features(pil_image_output, image_features_clip_model, preprocess).squeeze()
    clip_end_time = time.time()

    elapsed_time = clip_end_time - clip_start_time
    print(f"Time elapsed (clip): {elapsed_time:.4f} seconds")

    clip_output = torch.tensor(clip_output, dtype=torch.float32, requires_grad=True, device=DEVICE)
    loss = cosine_loss(clip_output, clip_target)
    print(f'Iteration #{i+1}, loss {loss.item()}')

    loss.backward()
    optimizer.step()
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Time elapsed: {elapsed_time:.4f} seconds")

image_output = sd.model.autoencoder_decode(random_latent)
pil_image_output = to_pil(image_output[0])

image_target = sd.model.autoencoder_decode(target_latent)
pil_image_target = to_pil(image_target[0])

output_dir = os.path.join('output', 'gradient_optimization')
os.makedirs(output_dir, exist_ok=True)
output_filepath = os.path.join(output_dir, 'clip_output.png')
target_filepath = os.path.join(output_dir, 'clip_target.png')

pil_image_target.save(target_filepath)
print(f"Target image saved to {target_filepath}")
pil_image_output.save(output_filepath)
print(f"Output image saved to {output_filepath}")
