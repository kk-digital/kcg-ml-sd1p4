import json
import argparse
import os
import sys
import random
from datetime import datetime
import time

base_directory = os.getcwd()
sys.path.insert(0, base_directory)
from scripts.text_to_image import Txt2Img
from stable_diffusion.model.unet.unet_attention import CrossAttention
from stable_diffusion.utils_image import save_images
from utility.labml import monit


def load_json(dataset_path):
    predictions = []
    # Opening JSON file
    f = open(dataset_path)

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    # Iterating through the json
    # list
    for item in data:
        predictions.append(item)

    # Closing file
    f.close()

    return predictions


def save_json(data, json_output):
    # save to json
    json_data = json.dumps(data, indent=4)

    # Save json
    with open(json_output, "w") as outfile:
        outfile.write(json_data)


# Sort by chad score starting from 0 to +INF
def create_folders_by_chad_score_range(predictions, output_path, num_class=10, method="uniform"):
    chad_scores = [prediction['chad-score-prediction'][0] for prediction in predictions]
    min_chad_score = min(chad_scores)
    max_chad_score = max(chad_scores)

    classes = []
    class_dict = {}

    if method == "top_percent":
        sorted_chad_scores = sorted(chad_scores, reverse=True)
        top_percentile_scores = sorted_chad_scores[:int(len(sorted_chad_scores) * num_class / 100)]
        
        # Find the smallest integer value in the top percentage (rounded down)
        min_range_value = int(min(top_percentile_scores))
        
        # Find the largest integer value in the top percentage (rounded up)
        max_range_value = int(max(top_percentile_scores)) + 1
        
        # Create folders from min_range_value to max_range_value
        for class_value in range(min_range_value, max_range_value):
            class_name = "{0}-{1}".format(class_value, class_value + 1)
    
            # create folder
            folder_name = os.path.join(output_path, class_name)
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
    
            classes.append(class_value)
            class_dict[class_value] = folder_name

    else:
        # Original computation for class range increment
        class_range_increment = round((max_chad_score - min_chad_score) / num_class)
        starting_class = round(min_chad_score) - 1

        class_value = starting_class

        # create folders
        for _ in range(num_class):
            class_name = "{0}-{1}".format(class_value, class_value + class_range_increment)

            # create folder
            folder_name = os.path.join(output_path, class_name)
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            classes.append(class_value)
            class_dict[class_value] = folder_name
            class_value += class_range_increment

    return classes, class_dict




def get_class_index(classes, score):
    len_classes = len(classes)
    for i in range(len_classes):
        if i == len_classes - 1:
            return i
        if classes[i] <= score <= classes[i + 1]:
            return i


def initialize_model(sampler, checkpoint_path, flash, steps,
                     force_cpu, cuda_device):
    # Set flash attention
    CrossAttention.use_flash_attention = flash

    # Starts the text2img
    txt2img = Txt2Img(
        sampler_name=sampler,
        n_steps=steps,
        force_cpu=force_cpu,
        cuda_device=cuda_device
    )
    txt2img.initialize_latent_diffusion(autoencoder=None, clip_text_embedder=None, unet_model=None,
                                        path=checkpoint_path, force_submodels_init=True)

    return txt2img


def generate_image(txt2img, prompt, negative_prompt, filename,
                   cfg_scale, low_vram, num_images, seed):
    prompts = [prompt]

    seed_string_array = str(seed)
    # Convert the elements in the list to integers (optional, if needed)
    seed_array = [int(num) for num in seed_string_array]

    with monit.section('Generate', total_steps=len(prompts)):
        for prompt in prompts:
            print(f'Generating images for prompt: "{prompt}"')

            for i in range(num_images):
                images = txt2img.generate_images(
                    batch_size=1,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    uncond_scale=cfg_scale,
                    low_vram=low_vram,
                    seed=seed_array[i % len(seed_array)]
                )

                save_images(images, filename)


def generate_image_using_prompt(predictions, classes, class_dict, limit_per_class, checkpoint_path):
    # default values
    negative_prompt = ""
    cfg_scale = 7
    sampler = 'ddim'
    flash = False
    steps = 20
    low_vram = False
    force_cpu = False
    cuda_device = "cuda:0"
    num_images = 1

    txt2img = initialize_model(sampler, checkpoint_path, flash, steps,
                               force_cpu, cuda_device)

    # initialize class_limit_dict
    class_limit_dict = {}
    for i in range(len(classes)):
        class_limit_dict[i] = 0

    for pred in predictions:
        class_index = get_class_index(classes, pred['chad-score-prediction'][0])

        # check if already in limit
        if class_limit_dict[class_index] >= limit_per_class:
            # skip
            continue

        class_folder_name = class_dict[classes[class_index]]

        prompt = pred['prompt-str']
        seed = random.randint(0, 2 ** 24)

        timestamp = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        image_filename = os.path.join(class_folder_name, f'{timestamp}-{i}.jpg')
        json_filename = os.path.join(class_folder_name, f'{timestamp}-{i}.json')

        # generate image
        generate_image(txt2img, prompt, negative_prompt, image_filename,
                       cfg_scale, low_vram, num_images, seed)

        # save json
        save_json(pred, json_filename)

        # update limit dict
        class_limit_dict[class_index] = class_limit_dict[class_index] + 1


def generate_image_based_on_classes(dataset_path, output_path, checkpoint_path, sampling_method="uniform", num_class=10, limit_per_class=128, top_k_percentage=10):
    predictions = load_json(dataset_path)
    print(len(predictions))
    
    #if sampling_method == "top_percent":
        # Sort and then slice the list to only retain the top K percentage of predictions
     #   predictions = sorted(predictions, key=lambda x: x['chad-score-prediction'][0], reverse=True)[:int(len(predictions) * top_k_percentage / 100)]
    if sampling_method == "top_percent":
        sorted_predictions = sorted(predictions, key=lambda x: x['chad-score-prediction'][0], reverse=True)
        slice_index = int(len(predictions) * top_k_percentage / 100)
        
        # Debug print statements
        print(f"Original length of predictions: {len(predictions)}")
        print(f"Top K% slice index: {slice_index}")
        print(f"Top prediction score: {sorted_predictions[0]['chad-score-prediction'][0]}")
        print(f"Score at slice index: {sorted_predictions[slice_index]['chad-score-prediction'][0]}")
        print(f"Chad scores of the top {top_k_percentage}%: {[pred['chad-score-prediction'][0] for pred in sorted_predictions[:slice_index]]}")
        
        predictions = sorted_predictions[:slice_index]
    
    elif sampling_method == "proportional_rejection":
        predictions = proportional_rejection_sampling(predictions)
    # If the sampling method is "uniform", we don't need to do anything additional, so no elif for it

    classes, class_dict = create_folders_by_chad_score_range(predictions, output_path, num_class, sampling_method)
    generate_image_using_prompt(predictions, classes, class_dict, limit_per_class, checkpoint_path)


def proportional_rejection_sampling(predictions, min_acceptance=5, max_acceptance=90):
    sampled_predictions = []
    min_score = min([pred['chad-score-prediction'][0] for pred in predictions])
    max_score = max([pred['chad-score-prediction'][0] for pred in predictions])

    for pred in predictions:
        score = pred['chad-score-prediction'][0]
        acceptance_prob = min_acceptance + (max_acceptance - min_acceptance) * ((score - min_score) / (max_score - min_score))
        if random.randint(0, 100) <= acceptance_prob:
            sampled_predictions.append(pred)

    return sampled_predictions


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate Image For Each Range Of Predicted Chad Score")

    parser.add_argument('--dataset-path', type=str, help='Predictions json path')
    parser.add_argument('--output-path', type=str, default="./output", help='Output path for the generated images')
    parser.add_argument('--checkpoint-path', type=str, default="./input/model/v1-5-pruned-emaonly.safetensors",
                        help='Path to the checkpoint file')
    parser.add_argument('--num-class', type=int, default=10, help='Number of classes to sort the images')
    parser.add_argument('--limit-per-class', type=int, default=128, help='Number of images to generate per class')
    parser.add_argument('--sampling-method', type=str, choices=["top_percent", "proportional_rejection", "uniform"], default="uniform",
                        help='Method of sampling from predictions. Options: top_percent, proportional_rejection, uniform')
    parser.add_argument('--top-k-percentage', type=int, default=10, help='Percentage of top scores to be considered when using top_percent sampling method.')

    return parser.parse_args()


def main():
    # Parser the parameters
    args = parse_arguments()
    start_time = time.time()

    # generate and save
    generate_image_based_on_classes(args.dataset_path, args.output_path, args.checkpoint_path, args.sampling_method, args.num_class,
                                    args.limit_per_class, args.top_k_percentage)

    print("Total Elapsed Time: {0}s".format(time.time() - start_time))



if __name__ == '__main__':
    main()
