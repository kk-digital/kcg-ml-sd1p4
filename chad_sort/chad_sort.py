import os
import shutil
import time
import zipfile
import torch
import sys
import json
import numpy as np

base_directory = os.getcwd()
sys.path.insert(0, base_directory)
sys.path.insert(0, os.path.join(base_directory, 'utils', 'dataset'))

from utility.dataset.image_dataset import ImageDataset
from chad_score.chad_score import ChadScorePredictor
from utility.utils_dirs import create_folder_if_not_exist

def sort_by_chad_score(dataset_path: str, device: str, num_classes: int, output_path: str):
    # Tracking time
    start_time = time.time()

    # Load the image dataset
    image_dataset = ImageDataset()
    image_dataset.load_dataset(dataset_path=dataset_path, is_tagged=False)

    chad_images = []
    min_chad_score = 999999.0
    max_chad_score = -999999.0

    chad_score_predictor = ChadScorePredictor(device=device)
    chad_score_predictor.load_model()

    for item in image_dataset.dataset:
        feature_vector_tensor = torch.tensor(item.feature_vector, device=device)
        chad_score = chad_score_predictor.get_chad_score(feature_vector_tensor)
        chad_image = ChadImage(os.path.basename(item.file_path), chad_score)
        chad_images.append(chad_image)

        min_chad_score = min(min_chad_score, chad_score)
        max_chad_score = max(max_chad_score, chad_score)

    current_image = 1
    num_images = len(chad_images)

    for chad_image in chad_images:
        zip_file_path = dataset_path
        image_filename = chad_image.file_path
        normalized_chad_score = (chad_image.chad_score - min_chad_score) / (max_chad_score - min_chad_score)
        class_index = (int)(normalized_chad_score * num_classes)

        # Open the zip file in read mode
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # Check if the 'images' folder exists in the zip file
            for item in zip_ref.namelist():
                if (item.endswith(image_filename)):
                    source = zip_ref.open(item)
                    dir = output_path + str(class_index)
                    create_folder_if_not_exist(dir)
                    image_path = dir + '/' + image_filename
                    target = open(image_path, "wb")
                    with source, target:
                        shutil.copyfileobj(source, target)
                        print("Saved image " + image_path)
                        print("Total images processed : " + str(current_image) + " out of " + str(num_images))
                        current_image += 1

    # Capture the ending time
    end_time = time.time()

    # Calculate the execution time
    execution_time = end_time - start_time

    print("Execution Time:", str(execution_time), "seconds")
    print("Images per second:", str(num_images / execution_time), " image/seconds")

def sort_dataset_by_chad_score(dataset_path: str, device: str, num_classes: int, output_path: str):
    # Tracking time
    start_time = time.time()

    data_list = []

    # Load the image dataset
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        file_paths = zip_ref.namelist()
        for file_path in file_paths:
            file_extension = os.path.splitext(file_path)[1]
            if file_extension == ".jpg":
                # get filename
                file_path_no_extension = os.path.splitext(file_path)[0]
                image_dir_path = os.path.dirname(file_path)
                root_path = os.path.dirname(image_dir_path)
                features_dir_path = os.path.join(root_path, "features")
                file_base_name = os.path.basename(file_path_no_extension)

                # get json
                file_path_json = file_path_no_extension + ".json"
                with zip_ref.open(file_path_json) as file:
                    json_content = json.load(file)

                # get embedding
                file_path_embedding = features_dir_path + '/' + file_base_name + ".embedding.npz"
                with zip_ref.open(file_path_embedding) as file:
                    embedding = np.load(file)
                    embedding_data = embedding['data']

                # get clip
                file_path_clip = features_dir_path + '/' + file_base_name + ".clip.npz"
                with zip_ref.open(file_path_clip) as file:
                    clip = np.load(file)
                    clip_data = clip['data']

                # get latent
                file_path_latent = features_dir_path + '/' + file_base_name + ".latent.npz";
                with zip_ref.open(file_path_latent) as file:
                    latent = np.load(file)
                    latent_data = latent['data']

                image_features = {
                    "file_path_json" : file_path_json,
                    "json_content" : json_content,
                    "embedding_data" : embedding_data,
                    "clip_data" : clip_data,
                    "latent_data" : latent_data,
                }

                # add image features to dataset
                data_list.append(image_features)

    images = []
    min_chad_score = 999999.0
    max_chad_score = -999999.0

    chad_score_predictor = ChadScorePredictor(device=device)
    chad_score_predictor.load_model()

    for item in data_list:
        chad_score = item['json_content']['chad_score']
        images.append(item)

        min_chad_score = min(min_chad_score, chad_score)
        max_chad_score = max(max_chad_score, chad_score)

    current_image = 1
    num_images = len(images)

    for image in images:
        zip_file_path = dataset_path
        image_filename = image['json_content']['image_name']
        file_path_json = image['file_path_json']
        normalized_chad_score = (image['json_content']['chad_score'] - min_chad_score) / (max_chad_score - min_chad_score)
        class_index = (int)(normalized_chad_score * num_classes)

        # output folders
        class_directory = output_path + str(class_index)
        images_directory = class_directory + '/images'
        features_directory = class_directory + '/features'

        # make sure the folders exist else create them
        create_folder_if_not_exist(class_directory)
        create_folder_if_not_exist(images_directory)
        create_folder_if_not_exist(features_directory)

        json_path = class_directory + '/' + file_path_json
        # Write JSON data to the file
        with open(json_path, "w") as json_file:
            json.dump(image['json_content'], json_file, indent=2)
            print("Saved json " + json_path)



        # Open the zip file in read mode
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # Check if the 'images' folder exists in the zip file
            for item in zip_ref.namelist():
                if (item.endswith(image_filename)):
                    source = zip_ref.open(item)
                    image_path = class_directory + '/' + image_filename
                    target = open(image_path, "wb")
                    with source, target:
                        shutil.copyfileobj(source, target)
                        print("Saved image " + image_path)
                        print("Total images processed : " + str(current_image) + " out of " + str(num_images))
                        current_image += 1

    # Capture the ending time
    end_time = time.time()

    # Calculate the execution time
    execution_time = end_time - start_time

    print("Execution Time:", str(execution_time), "seconds")
    print("Images per second:", str(num_images / execution_time), " image/seconds")