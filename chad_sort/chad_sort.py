import os
import shutil
import time
import zipfile
import torch
import sys

base_directory = os.getcwd()
sys.path.insert(0, base_directory)
sys.path.insert(0, os.path.join(base_directory, 'utils', 'dataset'))

from utility.dataset.image_dataset import ImageDataset
from chad_score.chad_score import ChadScorePredictor
from utility.utils_dirs import create_folder_if_not_exist

class ChadImage:
    def __init__(self, file_path, chad_score):
        self.file_path = file_path  # path + file name
        self.chad_score = chad_score


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
