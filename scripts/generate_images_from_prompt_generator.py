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


def create_if_doesnt_exist(path: str):
    # create folder if doesn't exist
    if not os.path.exists(path):
        os.mkdir(path)

def generate_images_from_prompt_generator(num_images, num_phrases, image_width, image_height, cfg_strength, batch_size,
                                          checkpoint_path, output, seed, flash, device, sampler, steps, force_cpu,
                                          num_datasets):
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
        print("Generating Dataset : " + str(current_task_index))
        folder_name = 'set_' + f'{current_task_index:04d}'
        output_path = os.path.join(output, folder_name)
        features_dir = os.path.join(output_path, 'features')
        image_dir = os.path.join(output_path, 'images')

        # create folder if doesn't exist
        create_if_doesnt_exist(output_path)
        create_if_doesnt_exist(features_dir)
        create_if_doesnt_exist(image_dir)

        generation_task_result_list = []
        for i in range(num_images):
            prompt_dict = generated_prompt_list[i]
            this_prompt = prompt_dict.prompt_str
            this_seed = seed_array[(i + current_task_index * num_images) % len(seed_array)]

            print("Generating image " + str(i) + " out of " + str(num_images));
            print("Prompt : ", this_prompt)
            print("Seed : ", this_seed)
            start_time = time.time()
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

            base_file_name = f'{i:04d}-{timestamp}'
            image_name = base_file_name + '.jpg'

            filename = os.path.join(image_dir, image_name)

            # Capture the starting time
            tmp_start_time = time.time()

            # no negative prompts for now
            negative_prompts = []
            un_cond, cond = txt2img.get_text_conditioning(cfg_strength, this_prompt, negative_prompts, batch_size)

            # Capture the ending time
            tmp_end_time = time.time()
            # Calculate the execution time
            tmp_execution_time = tmp_end_time - tmp_start_time

            print("Embedding vector Time:", tmp_execution_time, "seconds")

            # Capture the starting time
            tmp_start_time = time.time()

            latent = txt2img.generate_images_latent_from_embeddings(
                batch_size=batch_size,
                embedded_prompt=cond,
                null_prompt=un_cond,
                uncond_scale=cfg_strength,
                seed=this_seed,
                w=image_width,
                h=image_height
            )

            images =txt2img.get_image_from_latent(latent)

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
                cond_cpu = cond.cpu()

                cond = cond.detach()
                del cond
                torch.cuda.empty_cache()

                embedded_vector = cond_cpu.numpy()

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

            embedding_vector_filename = base_file_name + '.embedding.npz'
            clip_features_filename = base_file_name + '.clip.npz'
            latent_filename = base_file_name + '.latent.npz'

            generation_task_result = GenerationTaskResult(this_prompt, model_name, image_name,
                                                          embedding_vector_filename,
                                                          clip_features_filename, latent_filename,
                                                          image_hash, chad_score_model_name, chad_score, this_seed,
                                                          cfg_strength)
            # get numpy list from image_features
            with torch.no_grad():
                image_features_cpu = image_features.cpu()

                image_features = image_features.detach()
                del image_features
                torch.cuda.empty_cache()

                image_features_numpy = image_features_cpu.numpy()

            # save embedding vector to its own file
            embedding_vector_filepath = os.path.join(features_dir, embedding_vector_filename)
            np.savez_compressed(embedding_vector_filepath, data=embedded_vector)

            # save image features to its own file
            clip_features_filepath = os.path.join(features_dir, clip_features_filename)
            np.savez_compressed(clip_features_filepath, data=image_features_numpy)

            # save image latent to its own file
            latent_filepath = os.path.join(features_dir, latent_filename)
            np.savez_compressed(latent_filepath, data=latent)

            # save prompt dict to its own file
            prompt_dict_filename = base_file_name + '.prompt_dict.npz'
            prompt_dict_filepath = os.path.join(features_dir, prompt_dict_filename)
            np.savez_compressed(prompt_dict_filepath, prompt_dict=prompt_dict.prompt_dict,
                                prompt_str=prompt_dict.prompt_str, prompt_vector=prompt_dict.prompt_vector,
                                num_topics=prompt_dict.num_topics, num_modifiers=prompt_dict.num_modifiers,
                                num_styles=prompt_dict.num_styles,
                                num_constraints=prompt_dict.num_constraints)

            # Save the data to a JSON file
            json_filename = os.path.join(image_dir, base_file_name + '.json')

            generation_task_result_list.append({
                'image_filename': filename,
                'json_filename': json_filename,
                'embedding_vector_filepath': embedding_vector_filepath,
                'clip_features_filepath': clip_features_filepath,
                'latent_filepath': latent_filepath,
                'generation_task_result': generation_task_result
            })

            # save generation_task_result
            # save to json file
            generation_task_result.save_to_json(json_filename)

            # Capture the ending time
            end_time = time.time()

            # Calculate the execution time
            execution_time = end_time - start_time

            print("Execution Time:", execution_time, "seconds")

        # create zip for generated images
        shutil.make_archive(output_path, 'zip', output_path)
        print(f'Created zip file: {output_path}.zip')

        # Delete the entire set_000x folder after zipping
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
            print(f"Deleted folder: {output_path}")


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

    generate_images_from_prompt_generator(opt.num_images, opt.num_phrases, opt.image_width, opt.image_height,
                                          opt.cfg_scale,
                                          opt.batch_size, opt.checkpoint_path, opt.output, opt.seed, opt.flash,
                                          opt.cuda_device,
                                          opt.sampler, opt.steps, opt.force_cpu, opt.num_datasets)


if __name__ == "__main__":
    main()
