"""
---
title: Generate images using stable diffusion with a prompt
summary: >
 Generate images using stable diffusion with a prompt
---

# Generate images using [stable diffusion](../index.html) with a prompt
"""

import datetime
import os
import sys
import time
import zipfile
from zipfile import ZipFile
from random import randrange
import numpy as np
import torch
import random
import shutil
base_directory = "./"
sys.path.insert(0, base_directory)

from scripts.generate_images_from_random_prompt import Txt2Img
from ga.prompt_generator import *
from chad_score.chad_score import ChadScorePredictor
from model.util_clip import UtilClip
from prompt_generator import PromptGenerator
from generation_task_result import GenerationTaskResult
from stable_diffusion.utils_image import save_images
from stable_diffusion.model.unet.unet_attention import CrossAttention
from cli_builder import CLI

def generate_images_from_prompt_generator(num_images, num_phrases, image_width, image_height, cfg_strength, batch_size,
                                       checkpoint_path, output, seed, flash, device, sampler, steps, force_cpu, num_datasets):
    model_name = os.path.basename(checkpoint_path)
    # get prompts from prompt generator
    generated_prompt_list = generate_prompts(num_images, num_phrases)

    # Split the numbers_string into a list of substrings using the comma as the delimiter
    seed_string_array = []
    if seed != '':
        seed_string_array = seed.split(',')

    # default seed value is random int from 0 to 2^24
    if seed == '':
        # Generate an array of random integers in the range [0, 2^24)
        seed_string_array = [random.randint(0, 2 ** 24 - 1) for _ in range(num_images * num_datasets)]

    # Convert the elements in the list to integers
    seed_array = seed_string_array

    # Set flash attention
    CrossAttention.use_flash_attention = flash

    # Load default clip model
    util_clip = UtilClip(device=device)
    util_clip.load_model()

    # Load default chad model
    # hard coded for now
    chad_score_model_path = "input/model/chad_score/chad-score-v1.pth"
    chad_score_model_name = os.path.basename(chad_score_model_path)
    chad_score_predictor = ChadScorePredictor()
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

    for current_task_index in range(num_datasets):
        print("Generating Dataset : " +  str(current_task_index))
        folder_name = 'set_' + f'{current_task_index:04d}'
        output_path = os.path.join(output, folder_name)
        # create folder if doesn't exist
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        generation_task_result_list = []
        min_chad_score = 999999.0
        max_chad_score = -999999.0

        for i in range(num_images):
            this_prompt = generated_prompt_list[i].prompt_str
            this_seed = seed_array[(i + current_task_index * num_images) % len(seed_array)]

            print("Generating image " + str(i) + " out of " + str(num_images));
            print("Prompt : ", this_prompt)
            print("Seed : ", this_seed)
            start_time = time.time()
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

            base_file_name = f'{i:04d}-{timestamp}'
            image_name = base_file_name + '.jpg'

            filename = os.path.join(output_path, image_name)

            # Capture the starting time
            tmp_start_time = time.time()

            un_cond, cond = txt2img.get_text_conditioning(cfg_strength, this_prompt, batch_size)

            # Capture the ending time
            tmp_end_time = time.time()
            # Calculate the execution time
            tmp_execution_time = tmp_end_time - tmp_start_time

            print("Embedding vector Time:", tmp_execution_time, "seconds")

            # Capture the starting time
            tmp_start_time = time.time()

            images = txt2img.generate_images_from_embeddings(
                batch_size=batch_size,
                embedded_prompt=cond,
                null_prompt=un_cond,
                uncond_scale=cfg_strength,
                seed=this_seed,
                w=image_width,
                h=image_height
            )

            # Capture the ending time
            tmp_end_time = time.time()
            # Calculate the execution time
            tmp_execution_time = tmp_end_time - tmp_start_time

            print("Image generation Time:", tmp_execution_time, "seconds")

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
            image_features = util_clip.get_image_features(image)

            # Capture the ending time
            tmp_end_time = time.time()
            # Calculate the execution time
            tmp_execution_time = tmp_end_time - tmp_start_time

            print("Image features Time:", tmp_execution_time, "seconds")

            # Capture the starting time
            tmp_start_time = time.time()

            # compute chad score
            chad_score = chad_score_predictor.get_chad_score(image_features)

            # Capture the ending time
            tmp_end_time = time.time()
            # Calculate the execution time
            tmp_execution_time = tmp_end_time - tmp_start_time

            print("Chad Score Time:", tmp_execution_time, "seconds")

            # update the min, max for chad_score
            min_chad_score = min(min_chad_score, chad_score)
            max_chad_score = max(max_chad_score, chad_score)

            embedding_vector_filename = base_file_name + '.embedding.npz'
            clip_features_filename = base_file_name + '.clip.npz'
            latent_filename = base_file_name + '.latent.npz'

            generation_task_result = GenerationTaskResult(this_prompt, model_name, image_name, embedding_vector_filename,
                                                          clip_features_filename, latent_filename,
                                                          image_hash, chad_score_model_name, chad_score, this_seed,
                                                          cfg_strength)
            # get numpy list from image_features
            with torch.no_grad():
                image_features_numpy = image_features.cpu().numpy()

            # save embedding vector to its own file
            embedding_vector_filepath = output_path + '/' + embedding_vector_filename
            np.savez_compressed(embedding_vector_filepath, data=embedded_vector)

            # save image features to its own file
            clip_features_filepath = output_path + '/' + clip_features_filename
            np.savez_compressed(clip_features_filepath, data=image_features_numpy)

            # save image latent to its own file
            latent_filepath = output_path + '/' + latent_filename
            np.savez_compressed(latent_filepath, data=latent)

            # Save the data to a JSON file
            json_filename = output_path + '/' + base_file_name + '.json'

            # save generation_task_result
            # save to json file
            generation_task_result.save_to_json(json_filename)

            generation_task_result_list.append({
                'image_filename': filename,
                'json_filename': json_filename,
                'embedding_vector_filepath': embedding_vector_filepath,
                'clip_features_filepath': clip_features_filepath,
                'latent_filepath': latent_filepath,
                'generation_task_result': generation_task_result
            })

            # Capture the ending time
            end_time = time.time()

            # Calculate the execution time
            execution_time = end_time - start_time

            print("Execution Time:", execution_time, "seconds")

        # create zip for generated images
        shutil.make_archive(output_path, 'zip', output_path)


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
        .num_datasets() \
        .num_phrases() \
        .parse()

    generate_images_from_prompt_generator(opt.num_images, opt.num_phrases, opt.image_width, opt.image_height, opt.cfg_scale,
                                       opt.batch_size, opt.checkpoint_path, opt.output, opt.seed, opt.flash,
                                       opt.cuda_device,
                                       opt.sampler, opt.steps, opt.force_cpu, opt.num_datasets)


if __name__ == "__main__":
    main()