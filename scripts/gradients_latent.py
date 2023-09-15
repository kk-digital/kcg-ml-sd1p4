from PIL import Image

import argparse
import os
import sys
import numpy as np
import torch
import time
import shutil
import torch.nn as nn
import torch.optim as optim
import random
import torchvision
import zipfile
import torchvision.transforms as transforms

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())


from stable_diffusion.model_paths import SD_CHECKPOINT_PATH
from stable_diffusion.utils_backend import get_device
from ga.prompt_generator import generate_prompts
from stable_diffusion.model.clip_text_embedder import CLIPTextEmbedder
from stable_diffusion_base_script import StableDiffusionBaseScript
from stable_diffusion.utils_backend import get_autocast, set_seed
from chad_score.chad_score import ChadScorePredictor
from model.util_clip import UtilClip
from stable_diffusion.utils_image import save_images

def parse_arguments():
    """Command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Affine combination of embeddings.")

    parser.add_argument('--output', type=str, default='./output/gradients_latent')
    parser.add_argument('--image_width', type=int, default=512)
    parser.add_argument('--image_height', type=int, default=512)
    parser.add_argument('--cfg_strength', type=float, default=12)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--sampler', type=str, default='ddim')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--checkpoint_path',type=str, default=SD_CHECKPOINT_PATH)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_images', type=int, default=1)

    return parser.parse_args()



class Txt2Img(StableDiffusionBaseScript):
    """
    ### Text to image class
    """

    @torch.no_grad()
    def generate_images_latent_from_embeddings(self, *,
                                        seed: int = 0,
                                        batch_size: int = 1,
                                        embedded_prompt: torch.Tensor,
                                        null_prompt: torch.Tensor,
                                        h: int = 512, w: int = 512,
                                        uncond_scale: float = 7.5,
                                        low_vram: bool = False,
                                        noise_fn=torch.randn,
                                        temperature: float = 1.0,
                                        ):
        """
        :param seed: the seed to use when generating the images
        :param dest_path: is the path to store the generated images
        :param batch_size: is the number of images to generate in a batch
        :param prompt: is the prompt to generate images with
        :param h: is the height of the image
        :param w: is the width of the image
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param low_vram: whether to limit VRAM usage
        """
        # Number of channels in the image
        c = 4
        # Image to latent space resolution reduction
        f = 8

        if seed == 0:
            seed = time.time_ns() % 2 ** 32

        set_seed(seed)
        # Adjust batch size based on VRAM availability
        if low_vram:
            batch_size = 1

        # AMP auto casting
        autocast = get_autocast()
        with autocast:

            # [Sample in the latent space](../sampler/index.html).
            # `x` will be of shape `[batch_size, c, h / f, w / f]`
            x = self.sampler.sample(cond=embedded_prompt,
                                    shape=[batch_size, c, h // f, w // f],
                                    uncond_scale=uncond_scale,
                                    uncond_cond=null_prompt,
                                    noise_fn=noise_fn,
                                    temperature=temperature)

            return x



def compute_clip_similarity_cosine_distance(image_features, target_features):

    similarity = torch.nn.functional.cosine_similarity(image_features, target_features, dim=1, eps=1e-8)

    return similarity


def preprocess_image(images):
    image_mean = [0.48145466, 0.4578275, 0.40821073]
    image_std = [0.26862954, 0.26130258, 0.27577711]
    normalize = torchvision.transforms.Normalize(
        image_mean,
        image_std
    )
    resize = torchvision.transforms.Resize(224)
    center_crop = torchvision.transforms.CenterCrop(224)

    images = center_crop(images)
    images = resize(images)
    images = center_crop(images)
    images = normalize(images)

    return images

def latents_similarity_score(latent, index, output, target_features, device, save_image):

    images = txt2img.get_image_from_latent(latent)
    if save_image:
        image_list, image_hash_list = save_images(images, output + '/image' + str(index + 1) + '.jpg')

    # Map images to `[0, 1]` space and clip
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)

    images = preprocess_image(images)

    # Get the CLIP features
    image_features = util_clip.model.encode_image(images)

    image_features = image_features.to(torch.float32)

    fitness = compute_clip_similarity_cosine_distance(image_features, target_features)

    print("fitness : ", fitness.item())

    return fitness

def get_target_embeddings_features(util_clip, subject):

    features = util_clip.get_text_features(subject)
    features = features.to(torch.float32)

    return features

def log_to_file(message, output_directory):
    log_path = os.path.join(output_directory, "log.txt")

    with open(log_path, "a") as log_file:
        log_file.write(message + "\n")


if __name__ == "__main__":
    args = parse_arguments()

    output = args.output
    image_width = args.image_width
    image_height = args.image_height
    cfg_strength = args.cfg_strength
    device = get_device(args.device)
    steps = args.steps
    sampler = args.sampler
    checkpoint_path = args.checkpoint_path
    iterations = args.iterations
    learning_rate = args.learning_rate
    num_images = args.num_images

    starting_images_directory = output + '/' + 'starting_images'

    # Seed the random number generator with the current time
    random.seed(time.time())
    seed = 6789

    # make sure the directories are created
    os.makedirs(output, exist_ok=True)

    # Remove the directory and its contents recursively
    shutil.rmtree(output)

    # make sure the directories are created
    os.makedirs(output, exist_ok=True)
    os.makedirs(starting_images_directory, exist_ok=True)

    clip_text_embedder = CLIPTextEmbedder(device=get_device())
    clip_text_embedder.load_submodels()

    # Load default chad model
    # hard coded for now
    chad_score_model_path = "input/model/chad_score/chad-score-v1.pth"
    chad_score_model_name = os.path.basename(chad_score_model_path)
    chad_score_predictor = ChadScorePredictor(device=device)
    chad_score_predictor.load_model(chad_score_model_path)

    # Load the clip model
    util_clip = UtilClip(device=device)
    util_clip.load_model()

    # Starts the text2img
    txt2img = Txt2Img(
        sampler_name=sampler,
        n_steps=steps,
        force_cpu=False,
        cuda_device=device,
    )
    txt2img.initialize_latent_diffusion(autoencoder=None, clip_text_embedder=None, unet_model=None,
                                        path=checkpoint_path, force_submodels_init=True)

    # initial latent
    prompt_str = "accurate, short hair, (huge breasts), wide-eyed, (8k), witch, (((huge fangs))), sugar painting, 1girl, white box, lens flare, cowboy shot, full body, hairband, shirakami fubuki, photorealistic painting art by midjourney and greg rutkowski"
    negative_prompt_str = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck,  (low quality:2), normal quality, bad-hands-5, huge eyes, (variations:1.2), extra arms, extra fingers, lowres, edwigef, testicles, mutated, ((missing legs)), bad proportions, malformed limbs, low quality lowres black tongue, mutated hands, loli"


    embedded_prompts = clip_text_embedder(prompt_str)
    negative_embedded_prompts = clip_text_embedder(negative_prompt_str)

    latent = txt2img.generate_images_latent_from_embeddings(
        batch_size=1,
        embedded_prompt=embedded_prompts,
        null_prompt=negative_embedded_prompts,
        uncond_scale=cfg_strength,
        seed=seed,
        w=image_width,
        h=image_height
    )

    latent_tmp = latent
    latent = latent.clone()

    del latent_tmp

    images = txt2img.get_image_from_latent(latent)
    image_list, image_hash_list = save_images(images, starting_images_directory + '/image' + '.jpg')

    # Create a random latent tensor of shape (1, 4, 64, 64)
    #random_latent = torch.rand((1, 4, 64, 64), device=device, dtype=torch.float32)

    optimizer = optim.AdamW([latent], lr=learning_rate)
    mse_loss = nn.MSELoss()

    target = torch.tensor([1.0], device=device, dtype=torch.float32, requires_grad=True)

    start_time = time.time()

    fixed_taget_features = get_target_embeddings_features(util_clip, "chibi, anime, waifu, side scrolling")

    fixed_target = fixed_taget_features.detach()

    for i in range(0, iterations):
        # Zero the gradients

        save_image = True

        fitness = latents_similarity_score(latent, i, output, fixed_target, device, save_image)

        input = fitness
        loss = mse_loss(input, target)

        print(f'Iteration #{i + 1}, loss {loss}')
        print("grad : ", latent.grad)
        log_to_file(f'Iteration #{i + 1}, loss {loss}', output)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Time elapsed: {elapsed_time:.4f} seconds")
