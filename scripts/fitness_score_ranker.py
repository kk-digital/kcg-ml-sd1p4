import os
import zipfile
from PIL import Image
import shutil
import importlib.util
import argparse
import json


# Find the first callable in the module and return it
def find_fitness_function(fitness_module):
    for name in dir(fitness_module):
        obj = getattr(fitness_module, name)
        if callable(obj):
            return obj
    return None


def load_fitness_function(fitness_function_path):
    spec = importlib.util.spec_from_file_location("fitness_function", fitness_function_path)
    fitness_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fitness_module)
    fitness_function = find_fitness_function(fitness_module)
    if fitness_function is None:
        raise ValueError(f"No callable function found in {fitness_function_path}")
    return fitness_function


def test_images(fitness_function, fitness_function_filepath, zip_path, output_path):
    fitness_function_name = os.path.basename(fitness_function_filepath).replace('.py', '')

    folder_names = [f"{score / 10:.1f}" for score in range(11)]
    os.makedirs(output_path, exist_ok=True)

    for folder_name in folder_names:
        os.makedirs(os.path.join(output_path, folder_name), exist_ok=True)

    json_data = {}
    json_data["fitness_function_name"] = fitness_function_name
    json_data["images"] = {}

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('unzipped_images')

    for root, dirs, files in os.walk('unzipped_images'):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, filename)
                try:
                    pil_image = Image.open(img_path)
                    fitness_score = fitness_function(pil_image)
                    json_data["images"][filename] = fitness_score

                    category_score = int(fitness_score * 10)
                    if category_score == 10:
                        category_score = 9
                    category_folder = os.path.join(output_path, f"{category_score / 10:.1f}")

                    new_img_path = os.path.join(category_folder, filename)
                    pil_image.save(new_img_path)
                except Exception as e:
                    print(f"Failed to process file {img_path}. Error: {e}")

    sorted_images = {k: v for k, v in sorted(json_data["images"].items(), key=lambda item: item[1], reverse=True)}
    json_data["images"] = sorted_images

    json_file_path = os.path.join(output_path, 'fitness_scores.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    shutil.rmtree('unzipped_images')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test images and categorize by fitness score.")
    parser.add_argument('--fitness_function', type=str, required=True, help="Path to the Python file containing the fitness function.")
    parser.add_argument('--zip_path', type=str, required=True, help="Path to the ZIP file containing images.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to the output folder.")
    args = parser.parse_args()

    fitness_function = load_fitness_function(args.fitness_function)
    test_images(fitness_function, args.fitness_function, args.zip_path, args.output_path)
    print("Images ranked.")
