from PIL import Image

import argparse
import os
import sys
import numpy as np
import torch
import time

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from stable_diffusion.utils_backend import get_device
from ga.prompt_generator import generate_prompts
from stable_diffusion.model.clip_text_embedder import CLIPTextEmbedder
from stable_diffusion_base_script import StableDiffusionBaseScript
from stable_diffusion.utils_backend import get_autocast, set_seed

def parse_arguments():
    """Command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Affine combination of embeddings.")

    parser.add_argument('--output', type=str, help='Output folder')
    parser.add_argument('--image_width', type=int, default=64)
    parser.add_argument('--image_height', type=int, default=64)
    parser.add_argument('--cfg_strength', type=float, default=12)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--sampler', type=str, default='ddim')
    parser.add_argument('--num_prompts', type=int, default=12)
    parser.add_argument('--num_phrases', type=int, default=12)

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

def combine_embeddings(embeddings_array, weight_array):

    # empty embedding filled with zeroes
    result_embedding = np.zeros((1, 77, 768))

    num_elements = len(embeddings_array)
    for i in range(num_elements):
        embedding = embeddings_array[i]
        weight = weight_array[i]

        embedding = embedding * weight
        result_embedding += embedding

    return result_embedding

if __name__ == "__main__":
    args = parse_arguments()

    output = args.output
    image_width = args.image_width
    image_height = args.image_height
    cfg_strength = args.cfg_strength
    device = get_device(args.device)
    sampler = args.sampler
    num_prompts = args.num_prompts
    num_phrases = args.num_phrases

    clip_text_embedder = CLIPTextEmbedder(device=get_device())
    clip_text_embedder.load_submodels()

    # Load default chad model
    # hard coded for now
    chad_score_model_path = "input/model/chad_score/chad-score-v1.pth"
    chad_score_model_name = os.path.basename(chad_score_model_path)
    chad_score_predictor = ChadScorePredictor(device=device)
    chad_score_predictor.load_model(chad_score_model_path)

    # Starts the text2img
    txt2img = Txt2Img(
        sampler_name=sampler,
        n_steps=steps,
        force_cpu=force_cpu,
        cuda_device=device,
    )
    txt2img.initialize_latent_diffusion(autoencoder=None, clip_text_embedder=None, unet_model=None,
                                        path=checkpoint_path, force_submodels_init=True)


    # Generate N Prompts
    prompt_list = generate_prompts(num_prompts, num_phrases)

    embedded_prompts_array = []
    # Get N Embeddings
    for prompt in prompt_list:
        # get the embedding from positive text prompt
        embedded_prompts = clip_text_embedder(prompt.positive_prompt_str)
        embedded_prompts_numpy = embedded_prompts.detach().cpu().numpy()

        del embedded_prompts
        torch.cuda.empty_cache()

        embedded_prompts_array.append(embedded_prompts_numpy)

    # array of random weights
    weight_array = np.random.rand(num_prompts)
    # Combinate into one Embedding
    embedding_numpy = combine_embeddings(embedded_prompts_array, weight_array)
    # Maximize fitness

    embedding_vector = clip_text_embedder(embedding_numpy)
    latent = txt2img.generate_images_latent_from_embeddings(
                    batch_size=batch_size,
                    embedded_prompt=cond,
                    null_prompt=un_cond,
                    uncond_scale=cfg_strength,
                    seed=this_seed,
                    w=image_width,
                    h=image_height
                )