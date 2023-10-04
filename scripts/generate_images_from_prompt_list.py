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
from tqdm import tqdm

base_directory = "./"
sys.path.insert(0, base_directory)

from chad_score.chad_score import ChadScorePredictor
from model.util_clip import ClipOpenAi
from ga.prompt_generator import generate_prompts
from generation_task_result import GenerationTaskResult
from stable_diffusion.utils_backend import get_autocast, set_seed
from stable_diffusion.utils_image import save_images
from stable_diffusion_base_script import StableDiffusionBaseScript
from stable_diffusion.model.unet.unet_attention import CrossAttention
from utility.utils_argument_parsing import get_seed_array_from_string
from cli_builder import CLI
from utility.dataset.prompt_list_dataset import PromptListDataset


def get_batch_list(num_images, seed_array, current_task_index, image_dir, image_batch_size):
    batch_list = []
    current_batch = []
    current_batch_index = 0
    current_batch_image_index = 0
    # Loop through images and generate the prompt, seed for each one
    for i in range(num_images):
        print("Generating batches : image " + str(i) + " out of " + str(num_images));

        this_seed = seed_array[(i + current_task_index * num_images) % len(seed_array)]
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        total_digits = 4
        base_file_name = f'{i:0{total_digits}d}-{timestamp}'
        image_name = base_file_name + '.jpg'
        # Specify the filename with the path to the images directory under the respective set folder
        filename = os.path.join(image_dir, image_name)

        current_batch.append({
            'prompt_index': i,
            'seed': this_seed,
            "image_name": image_name,
            "filename": filename,
            "base_file_name": base_file_name
        })

        current_batch_image_index = current_batch_image_index + 1
        if current_batch_image_index >= image_batch_size or (i == (num_images - 1)):
            current_batch_image_index = 0
            batch_list.append(current_batch)
            current_batch = []
            current_batch_index += 1

    return batch_list


def get_embeddings(txt2img, prompt_dataset, batch, current_batch_index, image_batch_size, num_images, cfg_strength,
                   include_negative_prompt=False):
    # generate text embeddings in batches
    processed_images = current_batch_index * image_batch_size
    tmp_start_time = time.time()
    for task in batch:
        print("Get text embeddings " + str(processed_images + 1) + " out of " + str(num_images))
        processed_images = processed_images + 1

        prompt_index = task["prompt_index"]
        positive_prompt_str = prompt_dataset.get_prompt_data(prompt_index).positive_prompt_str
        negative_prompt_str = None

        if include_negative_prompt is True:
            negative_prompt_str = prompt_dataset.get_prompt_data(prompt_index).negative_prompt_str

        un_cond, cond = txt2img.get_text_conditioning(cfg_strength, positive_prompt_str, negative_prompt_str)
        task['cond'] = cond
        task['un_cond'] = un_cond
        del un_cond
        del cond

    tmp_end_time = time.time()
    tmp_execution_time = tmp_end_time - tmp_start_time
    print("Text embedding loading completed Time: {0:0.2f} seconds".format(tmp_execution_time))
    torch.cuda.empty_cache()

    return batch


def get_latents(batch, current_batch_index, image_batch_size, num_images, batch_size, cfg_strength, image_width,
                image_height, txt2img):
    # generating latent vector for each image
    processed_images = current_batch_index * image_batch_size
    tmp_start_time = time.time()
    for task in batch:
        print("Generating image latent " + str(processed_images + 1) + " out of " + str(num_images))
        processed_images = processed_images + 1

        cond = task['cond'].to(txt2img.device)

        un_cond = task['un_cond']
        if un_cond is not None:
            un_cond = un_cond.to(txt2img.device)

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

        task['latent'] = latent.cpu()
        del cond
        del un_cond
        del latent
        torch.cuda.empty_cache()

    tmp_end_time = time.time()
    tmp_execution_time = tmp_end_time - tmp_start_time
    print("Image latent generation completed time: {0:0.02f} seconds".format(tmp_execution_time))

    return batch


def generate_images_from_latents(batch, current_batch_index, image_batch_size, num_images, txt2img):
    # getting images from latent
    processed_images = current_batch_index * image_batch_size
    tmp_start_time = time.time()
    for task in batch:
        print("latent -> image " + str(processed_images + 1) + " out of " + str(num_images))
        processed_images = processed_images + 1

        latent = task['latent'].to(txt2img.device)
        filename = task['filename']

        images = txt2img.get_image_from_latent(latent)
        del latent

        image_list, image_hash_list = save_images(images, filename)
        del images
        torch.cuda.empty_cache()

        image_hash = image_hash_list[0]
        image = image_list[0]
        task['image'] = image
        task['image_hash'] = image_hash

    tmp_end_time = time.time()
    tmp_execution_time = tmp_end_time - tmp_start_time
    print("latent -> image completed time: {0:0.02f} seconds".format(tmp_execution_time))

    return batch


def compute_image_features(batch, current_batch_index, image_batch_size, num_images, util_clip):
    # compute image features
    processed_images = current_batch_index * image_batch_size
    tmp_start_time = time.time()
    for task in batch:
        print("Generating image features " + str(processed_images + 1) + " out of " + str(num_images))
        processed_images = processed_images + 1

        # get image features
        image = task['image']
        image_features = util_clip.get_image_features(image)
        task['image_features'] = image_features.cpu()
        del image
        del image_features
        torch.cuda.empty_cache()

    tmp_end_time = time.time()
    tmp_execution_time = tmp_end_time - tmp_start_time
    print("Image features generation completed time: {0:0.02f} seconds".format(tmp_execution_time))

    return batch


def compute_image_chad_score(batch, current_batch_index, image_batch_size, num_images, chad_score_predictor):
    # compute image chad_score
    processed_images = current_batch_index * image_batch_size
    tmp_start_time = time.time()
    for task in batch:
        print("Generating image chad score " + str(processed_images + 1) + " out of " + str(num_images))
        processed_images = processed_images + 1

        image_features = task['image_features'].to(chad_score_predictor.device)
        # compute chad_score
        chad_score = chad_score_predictor.get_chad_score(image_features)
        task['chad_score'] = chad_score
        del image_features
        torch.cuda.empty_cache()

    tmp_end_time = time.time()
    tmp_execution_time = tmp_end_time - tmp_start_time
    print("Image chad score completed time: {0:0.02f} seconds".format(tmp_execution_time))
    torch.cuda.empty_cache()

    return batch


def save_image_data(prompt_dataset, batch, current_batch_index, image_batch_size, num_images, feature_dir, image_dir,
                    model_name,
                    chad_score_model_name, cfg_strength, generation_task_result_list):
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

        # get prompt dict
        prompt_dict = prompt_dataset.get_prompt_data(task['prompt_index'], include_prompt_vector=True).to_json()

        image_name = task["image_name"]
        filename = task["filename"]
        image_hash = task["image_hash"]
        this_seed = task['seed']

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

        generation_task_result = GenerationTaskResult(prompt_dict["positive-prompt-str"], model_name, image_name,
                                                      embedding_vector_filename,
                                                      clip_features_filename, latent_filename,
                                                      image_hash, chad_score_model_name, chad_score, this_seed,
                                                      cfg_strength, negative_prompt=prompt_dict["negative-prompt-str"])
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
                            positive_prompt_str=prompt_dict["positive-prompt-str"],
                            negative_prompt_str=prompt_dict["negative-prompt-str"],
                            prompt_vector=prompt_dict["prompt-vector"],
                            num_topics=prompt_dict["num-topics"],
                            num_modifiers=prompt_dict["num-modifiers"],
                            num_styles=prompt_dict["num-styles"],
                            num_constraints=prompt_dict["num-constraints"])

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
        print("Saving of image data duration: {0:0.02f} seconds".format(tmp_execution_time))

    return generation_task_result_list


def generate_images_from_prompt_list(num_images,
                                     image_width,
                                     image_height,
                                     cfg_strength,
                                     batch_size,
                                     checkpoint_path,
                                     output,
                                     seed,
                                     flash,
                                     device,
                                     sampler,
                                     steps,
                                     force_cpu,
                                     num_datasets,
                                     image_batch_size,
                                     prompt_dataset,
                                     include_negative_prompt=False):
    
    
    model_name = os.path.basename(checkpoint_path)

    seed_array = get_seed_array_from_string(seed, array_size=(num_images * num_datasets))

    # Set flash attention
    CrossAttention.use_flash_attention = flash

    # Load default clip model
    util_clip = ClipOpenAi(device=device)
    util_clip.load_model()

    # Load default chad model
    # hard coded for now
    chad_score_model_path = "input/model/chad_score/chad-score-v1.pth"
    chad_score_model_name = os.path.basename(chad_score_model_path)
    chad_score_predictor = ChadScorePredictor(device=device)
    chad_score_predictor.load_model(chad_score_model_path)

    # Initialize text2img
    txt2img = StableDiffusionBaseScript(
        sampler_name=sampler,
        n_steps=steps,
        force_cpu=force_cpu,
        cuda_device=device,
    )
    txt2img.initialize_latent_diffusion(autoencoder=None, clip_text_embedder=None, unet_model=None,
                                        path=checkpoint_path, force_submodels_init=True)

    images_processed = 0
    zip_every_n = 10000  # Change this to your desired number

    for current_task_index in range(num_datasets):
        dataset_start_time = time.time()
        print("Generating Dataset : " + str(current_task_index))
        start_time = time.time()
        generation_task_result_list = []

        set_folder_name = f'set_{current_task_index:04}'
        feature_dir = os.path.join(output, set_folder_name, 'features')
        image_dir = os.path.join(output, set_folder_name, 'images')

        # make sure the directories are created
        os.makedirs(feature_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)

        batch_list = get_batch_list(num_images, seed_array, current_task_index, image_dir,
                                    image_batch_size)

        current_batch_index = 0
        for batch in tqdm(batch_list):
            batch_start_time = time.time()
            print("------ Batch " + str(current_batch_index + 1) + " out of " + str(len(batch_list)) + " ----------")

            batch = get_embeddings(txt2img, prompt_dataset, batch, current_batch_index, image_batch_size, num_images,
                                   cfg_strength, include_negative_prompt)
            batch = get_latents(batch, current_batch_index, image_batch_size, num_images, batch_size, cfg_strength,
                                image_width, image_height, txt2img)
            batch = generate_images_from_latents(batch, current_batch_index, image_batch_size, num_images, txt2img)
            batch = compute_image_features(batch, current_batch_index, image_batch_size, num_images, util_clip)
            batch = compute_image_chad_score(batch, current_batch_index, image_batch_size, num_images,
                                             chad_score_predictor)

            generation_task_result_list = save_image_data(prompt_dataset, batch, current_batch_index, image_batch_size,
                                                          num_images,
                                                          feature_dir, image_dir,
                                                          model_name, chad_score_model_name, cfg_strength,
                                                          generation_task_result_list)

            current_batch_index += 1
            batch_execution_time = time.time() - batch_start_time
            print("Batch duration: {0:0.02f} seconds".format(batch_execution_time))

            images_processed += len(batch) * image_batch_size
            if images_processed >= zip_every_n:
                # Zip the processed images
                set_directory_path = os.path.join(output, set_folder_name)
                zip_filename = os.path.join(output, f'{set_folder_name}_{images_processed}.zip')
                shutil.make_archive(zip_filename[:-4], 'zip', set_directory_path)
                print(f'Created zip file: {zip_filename}')

                # Reset the counter
                images_processed = 0

        for generation_task_result_item in generation_task_result_list:
            generation_task_result = generation_task_result_item['generation_task_result']
            json_filename = generation_task_result_item['json_filename']

            # save to json file
            generation_task_result.save_to_json(json_filename)

        # Zip the entire set folder
        set_directory_path = os.path.join(output, set_folder_name)
        zip_filename = os.path.join(output, f'{set_folder_name}.zip')
        shutil.make_archive(zip_filename[:-4], 'zip',
                            set_directory_path)  # The slice [:-4] removes ".zip" from the filename since make_archive appends it by default

        print(f'Created zip file: {zip_filename}')

        # Delete the entire set_000x folder after zipping
        if os.path.exists(set_directory_path):
            shutil.rmtree(set_directory_path)
            

        dataset_execution_time = time.time() - dataset_start_time
        print("Dataset generation duration: {0:0.02f} seconds".format(dataset_execution_time))


def main():
    opt = CLI('Generate images using stable diffusion with a prompt') \
        .batch_size() \
        .output() \
        .sampler() \
        .checkpoint_path() \
        .flash() \
        .steps() \
        .cfg_scale() \
        .force_cpu() \
        .cuda_device() \
        .num_images() \
        .seed() \
        .image_width() \
        .image_height() \
        .num_datasets() \
        .image_batch_size() \
        .prompt_list_dataset_path() \
        .include_negative_prompt() \
        .parse()

    num_images = opt.num_images

    # load prompt list
    limit = num_images
    prompt_dataset = PromptListDataset()
    prompt_dataset.load_prompt_list(opt.prompt_list_dataset_path, limit)

    # raise error when prompt list is not enough
    if len(prompt_dataset.prompt_paths) != num_images:
        raise Exception("Number of prompts do not match number of image to generate")

    # generate images
    generate_images_from_prompt_list(num_images,
                                     opt.image_width,
                                     opt.image_height,
                                     opt.cfg_scale,
                                     opt.batch_size,
                                     opt.checkpoint_path,
                                     opt.output,
                                     opt.seed,
                                     opt.flash,
                                     opt.cuda_device,
                                     opt.sampler,
                                     opt.steps,
                                     opt.force_cpu,
                                     opt.num_datasets,
                                     opt.image_batch_size,
                                     prompt_dataset,
                                     opt.include_negative_prompt)


if __name__ == "__main__":
    main()
