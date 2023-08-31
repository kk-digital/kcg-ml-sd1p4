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
import json

base_directory = "./"
sys.path.insert(0, base_directory)

from chad_score.chad_score import ChadScorePredictor
from model.util_clip import UtilClip
from ga.prompt_generator import generate_prompts
from generation_task_result import GenerationTaskResult
from stable_diffusion.utils_backend import get_autocast, set_seed
from stable_diffusion.utils_image import save_images
from stable_diffusion_base_script import StableDiffusionBaseScript
from stable_diffusion.model.unet.unet_attention import CrossAttention
from utility.utils_argument_parsing import get_seed_array_from_string
from cli_builder import CLI


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


def generate_images_from_prompt_generator(num_images, image_width, image_height, cfg_strength, batch_size,
                                       checkpoint_path, output, seed, flash, device, sampler, steps, force_cpu, num_datasets,
                                       image_batch_size, num_phrases):
    model_name = os.path.basename(checkpoint_path)

    seed_array = get_seed_array_from_string(seed, array_size=(num_images * num_datasets))

    # Set flash attention
    CrossAttention.use_flash_attention = flash

    # Load default clip model
    util_clip = UtilClip(device=device)
    util_clip.load_model()

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

    for current_task_index in range(num_datasets):
        print("Generating Dataset : " +  str(current_task_index))
        start_time = time.time()
        generation_task_result_list = []

        set_folder_name = f'set_{current_task_index:04}'
        feature_dir = os.path.join(output, set_folder_name, 'features')
        image_dir = os.path.join(output, set_folder_name, 'images')

        # make sure the directories are created
        os.makedirs(feature_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)

        prompt_list = generate_prompts(num_images, num_phrases)

        batch_list = []
        current_batch = []
        current_batch_index = 0
        current_batch_image_index = 0
        # Loop through images and generate the prompt, seed for each one
        for i in range(num_images):
            print("Generating batches : image " + str(i) + " out of " + str(num_images));

            this_prompt = prompt_list[i].positive_prompt_str
            prompt_dict = prompt_list[i]
            this_seed = seed_array[(i + current_task_index * num_images) % len(seed_array)]
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            total_digits = 4
            base_file_name = f'{i:0{total_digits}d}-{timestamp}'
            image_name = base_file_name + '.jpg'
            # Specify the filename with the path to the images directory under the respective set folder
            filename = os.path.join(image_dir, image_name)

            current_batch.append({
                'prompt' : this_prompt,
                'prompt_dict' : prompt_dict,
                'seed' : this_seed,
                "image_name" : image_name,
                "filename" : filename,
                "base_file_name" : base_file_name
            })

            current_batch_image_index = current_batch_image_index + 1
            if current_batch_image_index >= image_batch_size or (i == (num_images - 1)):
                current_batch_image_index = 0
                batch_list.append(current_batch)
                current_batch = []
                current_batch_index += 1


        current_batch_index = 0
        for batch in batch_list:
            print("------ Batch " + str(current_batch_index + 1) + " out of " + str(len(batch_list)) + " ----------")
            #print(torch.cuda.memory_allocated() / 1024.0 / 1024.0)
            # generate text embeddings in batches
            processed_images = current_batch_index * image_batch_size
            tmp_start_time = time.time()
            for task in batch:
                print("Generating text embeddings " + str(processed_images + 1) + " out of " + str(num_images))
                processed_images = processed_images + 1

                this_prompt = task['prompt']
                # no negative prompts for now
                negative_prompts = []
                un_cond, cond = txt2img.get_text_conditioning(cfg_strength, this_prompt, negative_prompts, batch_size)
                task['cond'] = cond
                task['un_cond'] = un_cond

            tmp_end_time = time.time()
            tmp_execution_time = tmp_end_time - tmp_start_time
            print("Text embedding generation completed Time:", tmp_execution_time, "seconds")

            # generating latent vector for each image
            processed_images = current_batch_index * image_batch_size
            tmp_start_time = time.time()
            for task in batch:
                print("Generating image latent " + str(processed_images + 1) + " out of " + str(num_images))
                processed_images = processed_images + 1

                cond = task['cond']
                un_cond = task['un_cond']
                this_seed = task['seed']

                latent = txt2img.generate_images_latent_from_embeddings(
                    batch_size=batch_size,
                    embedded_prompt=cond,
                    null_prompt=un_cond,
                    uncond_scale=cfg_strength,
                    seed=this_seed,
                    w=image_width,
                    h=image_height
                )

                task['latent'] = latent

            tmp_end_time = time.time()
            tmp_execution_time = tmp_end_time - tmp_start_time
            print("Image latent generation completed Time:", tmp_execution_time, "seconds")

            # getting images from latent
            processed_images = current_batch_index * image_batch_size
            tmp_start_time = time.time()
            for task in batch:
                print("latent -> image " + str(processed_images + 1) + " out of " + str(num_images))
                processed_images = processed_images + 1

                latent = task['latent']
                filename = task['filename']

                images = txt2img.get_image_from_latent(latent)
                image_list, image_hash_list = save_images(images, filename)
                image_hash = image_hash_list[0]
                image = image_list[0]
                task['image'] = image
                task['image_hash'] = image_hash

            tmp_end_time = time.time()
            tmp_execution_time = tmp_end_time - tmp_start_time
            print("latent -> image completed Time:", tmp_execution_time, "seconds")

            # compute image features
            processed_images = current_batch_index * image_batch_size
            tmp_start_time = time.time()
            for task in batch:
                print("Generating image features " + str(processed_images + 1) + " out of " + str(num_images))
                processed_images = processed_images + 1

                # get image features
                image = task['image']
                image_features = util_clip.get_image_features(image)
                task['image_features'] = image_features

            tmp_end_time = time.time()
            tmp_execution_time = tmp_end_time - tmp_start_time
            print("Image features generation completed Time:", tmp_execution_time, "seconds")

            # compute image chad_score
            processed_images = current_batch_index * image_batch_size
            tmp_start_time = time.time()
            for task in batch:
                print("Generating image chad score " + str(processed_images + 1) + " out of " + str(num_images))
                processed_images = processed_images + 1

                image_features = task['image_features']
                # compute chad_score
                chad_score = chad_score_predictor.get_chad_score(image_features)
                task['chad_score'] = chad_score

            tmp_end_time = time.time()
            tmp_execution_time = tmp_end_time - tmp_start_time
            print("Image chad score completed Time:", tmp_execution_time, "seconds")

            # compute image chad_score
            processed_images = current_batch_index * image_batch_size
            tmp_start_time = time.time()
            for task in batch:
                print("Image ready to be zipped " + str(processed_images + 1) + " out of " + str(num_images))
                processed_images = processed_images + 1

                cond = task['cond']
                un_cond = task['un_cond']
                latent = task['latent']
                image_features = task['image_features']
                base_file_name = task['base_file_name']
                chad_score = task['chad_score']
                prompt_dict = task['prompt_dict']

                image_name = task["image_name"]
                filename = task["filename"]

                # convert tensor to numpy array
                with torch.no_grad():
                    embedded_vector = cond.cpu().numpy()
                # free cond memory

                del task['cond']
                torch.cuda.empty_cache()
                # free un_cond memory

                del task['un_cond']
                torch.cuda.empty_cache()
                # image latent
                with torch.no_grad():
                    latent = latent.cpu().numpy()

                del task['latent']
                torch.cuda.empty_cache()

                embedding_vector_filename = base_file_name + '.embedding.npz'
                clip_features_filename = base_file_name + '.clip.npz'
                latent_filename = base_file_name + '.latent.npz'
                prompt_dict_filename = base_file_name + '.prompt_dict.npz'

                embedding_vector_filepath = feature_dir + '/' + embedding_vector_filename
                clip_features_filepath = feature_dir + '/' + clip_features_filename
                latent_filepath = feature_dir + '/' + latent_filename
                prompt_dict_filepath = feature_dir + '/' + prompt_dict_filename
                json_filename = os.path.join(image_dir, base_file_name + '.json')

                generation_task_result = GenerationTaskResult(this_prompt, model_name, image_name,
                                                              embedding_vector_filename,
                                                              clip_features_filename, latent_filename,
                                                              image_hash, chad_score_model_name, chad_score, this_seed,
                                                              cfg_strength)
                # get numpy list from image_features
                with torch.no_grad():
                    image_features_numpy = image_features.cpu().numpy()

                # free image_features memory
                del task['image_features']
                torch.cuda.empty_cache()

                # save embedding vector to its own file
                np.savez_compressed(embedding_vector_filepath, data=embedded_vector)
                # save image features to its own file
                np.savez_compressed(clip_features_filepath, data=image_features_numpy)
                # save image latent to its own file
                np.savez_compressed(latent_filepath, data=latent)
                # save prompt dictionary to its own file
                np.savez_compressed(prompt_dict_filepath,
                                    positive_prompt_str=prompt_dict.positive_prompt_str,
                                    negative_prompt_str=prompt_dict.negative_prompt_str,
                                    prompt_vector=prompt_dict.prompt_vector,
                                    num_topics=prompt_dict.num_topics,
                                    num_modifiers=prompt_dict.num_modifiers,
                                    num_styles=prompt_dict.num_styles,
                                    num_constraints=prompt_dict.num_constraints)

                # Save the data to a JSON file
                with open(json_filename, 'w') as json_file:
                    json.dump(generation_task_result.to_dict(), json_file)

                generation_task_result_list.append({
                    'image_filename': filename,
                    'json_filename': json_filename,
                    'embedding_vector_filepath': embedding_vector_filepath,
                    'clip_features_filepath': clip_features_filepath,
                    'latent_filepath': latent_filepath,
                    'generation_task_result': generation_task_result
                })


            tmp_end_time = time.time()
            tmp_execution_time = tmp_end_time - tmp_start_time
            current_batch_index = current_batch_index + 1

        # chad score value should be between [0, 1]
        for generation_task_result_item in generation_task_result_list:
            generation_task_result = generation_task_result_item['generation_task_result']
            json_filename = generation_task_result_item['json_filename']

            # save to json file
            generation_task_result.save_to_json(json_filename)



        # Zip the entire set folder
        set_directory_path = os.path.join(output, set_folder_name)
        zip_filename = os.path.join(output, f'{set_folder_name}.zip')
        shutil.make_archive(zip_filename[:-4], 'zip', set_directory_path)  # The slice [:-4] removes ".zip" from the filename since make_archive appends it by default

        print(f'Created zip file: {zip_filename}')

        # Delete the entire set_000x folder after zipping
        if os.path.exists(set_directory_path):
            shutil.rmtree(set_directory_path)
            print(f"Deleted folder: {set_folder_name}")




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
        .image_batch_size() \
        .num_phrases() \
        .parse()

    generate_images_from_prompt_generator(opt.num_images, opt.image_width, opt.image_height, opt.cfg_scale,
                                       opt.batch_size, opt.checkpoint_path, opt.output, opt.seed, opt.flash,
                                       opt.cuda_device,
                                       opt.sampler, opt.steps, opt.force_cpu, opt.num_datasets, opt.image_batch_size,
                                       opt.num_phrases)


if __name__ == "__main__":
    main()
