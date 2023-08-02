"""
---
title: Generate images using stable diffusion with a prompt
summary: >
 Generate images using stable diffusion with a prompt
---

# Generate images using [stable diffusion](../index.html) with a prompt
"""

import time
import os
import sys
from datetime import datetime
from zipfile import ZipFile
import random
import torch
import clip
import numpy as np

base_directory = "./"
sys.path.insert(0, base_directory)

from utils.clip.clip_features_image import ClipImageFeatures
from prompt_generator import PromptGenerator
from generation_task_result import GenerationTaskResult
from stable_diffusion.utils_backend import get_autocast, set_seed
from stable_diffusion.utils_image import save_images
from stable_diffusion_base_script import StableDiffusionBaseScript
from utility.labml import monit
from stable_diffusion.model.unet.unet_attention import CrossAttention
from cli_builder import CLI
from chad_score.chad_score import get_chad_score

class Txt2Img(StableDiffusionBaseScript):
    """
    ### Text to image class
    """

    @torch.no_grad()
    def generate_images(self, *,
                        seed: int = 0,
                        batch_size: int = 1,
                        prompt: str,
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

        # Make a batch of prompts
        prompts = batch_size * [prompt]

        # AMP auto casting
        autocast = get_autocast()
        with autocast:
            un_cond, cond = self.get_text_conditioning(uncond_scale, prompts, batch_size)

            start_time = time.time()
            # [Sample in the latent space](../sampler/index.html).
            # `x` will be of shape `[batch_size, c, h / f, w / f]`
            x = self.sampler.sample(cond=cond,
                                    shape=[batch_size, c, h // f, w // f],
                                    uncond_scale=uncond_scale,
                                    uncond_cond=un_cond,
                                    noise_fn=noise_fn,
                                    temperature=temperature)

            # Capture the ending time
            end_time = time.time()

            # Calculate the execution time
            execution_time = end_time - start_time

            print("Sampling Time:", execution_time, "seconds")

            return self.decode_image(x)

    @torch.no_grad()
    def generate_images_from_embeddings(self, *,
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

            return self.decode_image(x)


def main():
    opt = CLI('Generate images using stable diffusion with a prompt') \
        .prompts_file(check_exists=True, required=False) \
        .batch_size() \
        .output() \
        .sampler() \
        .checkpoint_path() \
        .flash() \
        .steps() \
        .cfg_scale() \
        .low_vram() \
        .force_cpu() \
        .cuda_device() \
        .num_images() \
        .seed() \
        .output_metadata() \
        .image_width() \
        .image_height() \
        .parse()

    # Hard coded prompts
    arg_prompt = r"chibi, waifu, scifi, side scrolling, character, side scrolling, white background, centered," \
             r" full character, no background, not centered, line drawing, sketch, black and white," \
             r" colored, offset, video game,exotic, sureal, miltech, fantasy, frank frazetta," \
             r" terraria, final fantasy, cortex command, surreal, water color expressionist, david mckean, " \
             r" jock, esad ribic, chris bachalo, expressionism, Jackson Pollock, Alex Kanevskyg, Francis Bacon, Trash Polka," \
             r" abstract realism, andrew salgado, alla prima technique, alla prima, expressionist alla prima, expressionist alla prima technique"

    prompt = arg_prompt

    image_width = opt.image_width
    image_height = opt.image_height
    model_name = os.path.basename(opt.checkpoint_path)
    # Split the numbers_string into a list of substrings using the comma as the delimiter
    seed_string_array = []
    if opt.seed != '':
        seed_string_array = opt.seed.split(',')

    # Convert the elements in the list to integers
    seed_array = [int(num) for num in seed_string_array]

    if len(seed_array) == 0:
        seed_array = [random.randint(0, 2**31-1) for _ in range(opt.num_images)]

    # Set flash attention
    CrossAttention.use_flash_attention = opt.flash

    # Load default clip model
    clip_image_features = ClipImageFeatures(opt.cuda_device)
    clip_image_features.load_model()

    # Starts the text2img
    txt2img = Txt2Img(
        sampler_name=opt.sampler,
        n_steps=opt.steps,
        force_cpu=opt.force_cpu,
        cuda_device=opt.cuda_device,
    )
    txt2img.initialize_latent_diffusion(autoencoder=None, clip_text_embedder=None, unet_model=None,
                                        path=opt.checkpoint_path, force_submodels_init=True)

    current_task_index = 0
    generation_task_result_list = []
    min_chad_score = 999999.0
    max_chad_score = -999999.0

    prompt_list = prompt.split(',');
    prompt_generator = PromptGenerator(prompt_list)

    for i in range(opt.num_images):

        num_prompts_per_image = 12
        this_prompt = prompt_generator.random_prompt(num_prompts_per_image)
        this_seed = seed_array[i % len(seed_array)]

        print("Generating image " + str(i) + " out of " + str(opt.num_images));
        print("Prompt : ", this_prompt)
        print("Seed : ", this_seed)
        start_time = time.time()
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        total_digits = 4

        base_file_name =  f'{i:0{total_digits}d}-{timestamp}'
        image_name = base_file_name + '.jpg'

        filename = opt.output + '/' + image_name


        # Capture the starting time
        tmp_start_time = time.time()

        un_cond, cond = txt2img.get_text_conditioning(opt.cfg_scale, this_prompt, opt.batch_size)

        # Capture the ending time
        tmp_end_time = time.time()
        # Calculate the execution time
        tmp_execution_time = tmp_end_time - tmp_start_time

        print("Embedding vector Time:", tmp_execution_time, "seconds")

        images = txt2img.generate_images_from_embeddings(
            batch_size=opt.batch_size,
            embedded_prompt=cond,
            null_prompt=un_cond,
            uncond_scale=opt.cfg_scale,
            low_vram=opt.low_vram,
            seed=this_seed,
            w = image_width,
            h = image_height
        )

        image_list, image_hash_list = save_images(images, filename)
        image_hash = image_hash_list[0]
        image = image_list[0]

        # convert tensor to numpy array
        with torch.no_grad():
            embedded_vector = cond.cpu().numpy()


        # image latent
        latent = []

        # Capture the starting time
        tmp_start_time = time.time()

        # get image features
        image_features = clip_image_features.get_image_features(image)

        # Capture the ending time
        tmp_end_time = time.time()
        # Calculate the execution time
        tmp_execution_time = tmp_end_time - tmp_start_time

        print("Image features Time:", tmp_execution_time, "seconds")

        # hard coded for now
        chad_score_model_path = "input/model/chad_score/chad-score-v1.pth"
        chad_score_model_name = os.path.basename(chad_score_model_path)

        # Capture the starting time
        tmp_start_time = time.time()

        # compute chad score
        chad_score = get_chad_score(image_features, chad_score_model_path, device=opt.cuda_device)

        # Capture the ending time
        tmp_end_time = time.time()
        # Calculate the execution time
        tmp_execution_time = tmp_end_time - tmp_start_time

        print("Chad Score Time:", tmp_execution_time, "seconds")

        # update the min, max for chad_score
        min_chad_score = min(min_chad_score, chad_score.item())
        max_chad_score = max(max_chad_score, chad_score.item())

        embedding_vector_filename = base_file_name + '.embedding.npz'
        clip_features_filename =  base_file_name + '.clip.npz'
        latent_filename = base_file_name + '.latent.npz'

        generation_task_result = GenerationTaskResult(this_prompt, model_name, image_name, embedding_vector_filename, clip_features_filename,latent_filename,
                                                     image_hash, chad_score_model_name, chad_score.item(), this_seed, opt.cfg_scale)
        # get numpy list from image_features
        with torch.no_grad():
            image_features_numpy = image_features.cpu().numpy()


        # save embedding vector to its own file
        embedding_vector_filepath = opt.output + '/' + embedding_vector_filename
        np.savez_compressed(embedding_vector_filepath, data=embedded_vector)

        # save image features to its own file
        clip_features_filepath = opt.output + '/' + clip_features_filename
        np.savez_compressed(clip_features_filepath, data=image_features_numpy)

        # save image latent to its own file
        latent_filepath = opt.output + '/' + latent_filename
        np.savez_compressed(latent_filepath, data=latent)

        # Save the data to a JSON file
        json_filename = opt.output + '/' + base_file_name + '.json'


        generation_task_result_list.append({
            'image_filename': filename,
            'json_filename' : json_filename,
            'embedding_vector_filepath': embedding_vector_filepath,
            'clip_features_filepath' : clip_features_filepath,
            'latent_filepath' : latent_filepath,
            'generation_task_result' : generation_task_result
        })

        # Capture the ending time
        end_time = time.time()

        # Calculate the execution time
        execution_time = end_time - start_time

        print("Execution Time:", execution_time, "seconds")

    # chad score value should be between [0, 1]
    for generation_task_result_item in generation_task_result_list:
        generation_task_result = generation_task_result_item['generation_task_result']
        json_filename = generation_task_result_item['json_filename']

        # chad score value should be between [0, 1]
        normalized_chad_score = (generation_task_result.chad_score - min_chad_score) / (max_chad_score - min_chad_score)
        generation_task_result.chad_score = normalized_chad_score

        # save to json file
        generation_task_result.save_to_json(json_filename)

    total_digits = 4

    zip_filename = opt.output + '/' + 'set_' + f'{current_task_index:0{total_digits}d}' + '.zip';
    # create zip for generated images
    with ZipFile(zip_filename, 'w') as file:
        print('Created zip file ' + zip_filename)
        zip_task_index = 1
        for generation_task_result_item in generation_task_result_list:
            print('Zipping task ' + str(zip_task_index) + ' out of ' + str(len(generation_task_result_list)))
            generation_task_result = generation_task_result_item['generation_task_result']

            json_filename = generation_task_result_item['json_filename']
            image_filename = generation_task_result_item['image_filename']
            embedding_vector_filename = generation_task_result.embedding_name
            clip_features_filename = generation_task_result.clip_name
            latent_filename = generation_task_result.latent_name

            embedding_vector_filepath = generation_task_result_item['embedding_vector_filepath']
            clip_features_filepath = generation_task_result_item['clip_features_filepath']
            latent_filepath = generation_task_result_item['latent_filepath']

            file.write(json_filename, arcname=os.path.basename(json_filename))
            file.write(image_filename, arcname=os.path.basename(image_filename))
            file.write(embedding_vector_filepath, arcname=embedding_vector_filename)
            file.write(clip_features_filepath, arcname=clip_features_filename)
            file.write(latent_filepath, arcname=latent_filename)
            zip_task_index += 1

if __name__ == "__main__":
    main()
