import argparse
import os
import shutil
import sys

base_directory = os.getcwd()
sys.path.insert(0, base_directory)
sys.path.insert(0, os.path.join(base_directory, 'utils', 'dataset'))

from chad_sort.chad_sort import sort_by_chad_score
from stable_diffusion.utils_backend import get_device
from utility.utils_dirs import remove_all_files_and_folders


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

    # sort by chad score
    # sort_by_chad_score(dataset_path, device, num_classes, output_path)
    sort_dataset_by_chad_score(dataset_path, device, num_classes, output_path)


if __name__ == '__main__':
    main()
