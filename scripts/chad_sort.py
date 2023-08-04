import argparse
import os
import shutil
import sys
import time
import zipfile

import torch


base_directory = os.getcwd()
sys.path.insert(0, base_directory)
sys.path.insert(0, os.path.join(base_directory, 'utils', 'dataset'))

from stable_diffusion.utils_backend import get_device
from utility.utils.dataset.image_dataset import ImageDataset
from chad_score.chad_score import ChadScorePredictor


def parse_arguments():
    """Command-line arguments for 'classify' command."""
    parser = argparse.ArgumentParser(
        description="Chad sort, takes in an image database and sorts it by chad score into many folders.")

    parser.add_argument('--dataset-path', type=str, help='Path to the dataset to sort')
    parser.add_argument('--output-path', type=str, default='./output/chad_sort/', help='Path to the output folder')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='Defines the number of classes to sort into, Specifies the total count of categories or groups or folders')
    parser.add_argument('--device', type=str, default=None, help='Path to the dataset to sort')

    return parser.parse_args()


def ensure_required_args(args):
    """Check if required arguments are set."""

    if not args.dataset_path:
        print('Error: --dataset_path is required')
        sys.exit(1)


def create_folder_if_not_exist(folder_path):
    """
    Create a folder if it does not exist.

    Parameters:
        folder_path (str): The path of the folder to create.

    Returns:
        None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def remove_all_files_and_folders(path):
    """
    Remove all files and folders inside the specified path.

    Parameters:
        path (str): The path for which to remove all files and folders.

    Returns:
        None
    """
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            shutil.rmtree(dir_path)

    print(f"All files and folders inside '{path}' have been removed.")


class ChadImage:
    def __init__(self, file_path, chad_score):
        self.file_path = file_path  # path + file name
        self.chad_score = chad_score


def main():
    # Parser the parameters
    args = parse_arguments()

    dataset_path = args.dataset_path
    num_classes = args.num_classes
    output_path = args.output_path
    device = get_device(args.device)

    # Make sure we the user provides the required arguments
    ensure_required_args(args)

    # make sure the output path exists
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Clean the output folder
    remove_all_files_and_folders(output_path)

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


if __name__ == '__main__':
    main()
