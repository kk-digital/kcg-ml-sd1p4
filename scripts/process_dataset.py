import sys
import os
import argparse
from tqdm import tqdm
sys.path.insert(0, os.getcwd())
from utility.dataset.image_dataset_storage_format.validator import ImageDatasetStorageFormatValidator
from utility.dataset.image_dataset_storage_format.processor import ImageDatasetStorageFormatProcessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Add sub-commands
    # dataset-validate subcommand
    dataset_validate_parser = subparsers.add_parser("dataset-validate", help="...",
                                                           description="CLI tool for validating image dataset storage format")
    dataset_validate_parser.add_argument("--image-dataset-path", help="The path of the image dataset to validate")
    dataset_validate_parser.add_argument("--is-tagged", type=bool, default=False, help="True if dataset is a tagged dataset (default=False)")

    # format-and-compute-manifest subcommand
    format_compute_manifest_parser = subparsers.add_parser("format-and-compute-manifest", help="...", description="CLI tool for formatting images and computing manifest of dataset")
    format_compute_manifest_parser.add_argument("--image-dataset-path", type=str, help="The path of the image dataset to process")
    format_compute_manifest_parser.add_argument("--is-tagged", type=bool, default=False, help="True if dataset is a tagged dataset (default=False)")
    format_compute_manifest_parser.add_argument("--output-path", type=str, default="./output", help="The path to save the dataset (default='./output')")

    # compute-features-of-zip command
    compute_features_of_zip_parser = subparsers.add_parser("compute-features-of-zip", help="...",
                                                           description="CLI tool for computing features of dataset")
    compute_features_of_zip_parser.add_argument("--image-dataset-path", type=str, help="The path of the image dataset to process")
    compute_features_of_zip_parser.add_argument("--clip-model", type=str, default="ViT-L/14", help="The CLIP model to use for computing features (default='ViT-L/14')")
    compute_features_of_zip_parser.add_argument("--batch-size", type=int, default=8, help="The batch size to use (default=8)")

    args = parser.parse_args()

    # check if dataset path given is a dir
    dataset_paths = []
    if os.path.isdir(args.image_dataset_path):
        for root, _, files in os.walk(args.image_dataset_path):
            for file in files:
                if file.endswith(".zip"):
                    dataset_paths.append(os.path.join(root, file))
        print("Found {0} zip files to process.".format(len(dataset_paths)))

    if args.command == "dataset-validate":
        validator = ImageDatasetStorageFormatValidator()

        if dataset_paths:
            for path in tqdm(dataset_paths):
                print("-------------------------------")
                print("Validating {0}...".format(path))
                validator.validate_dataset(path, args.is_tagged)
            print("Finished Validating {0} zip files.".format(len(dataset_paths)))
        else:
            validator.validate_dataset(args.image_dataset_path, args.is_tagged)

    elif args.command == "format-and-compute-manifest":
        processor = ImageDatasetStorageFormatProcessor()

        if dataset_paths:
            for path in tqdm(dataset_paths):
                print("-------------------------------")
                print("Processing {0}...".format(path))
                processor.format_and_compute_manifest(path, args.is_tagged, args.output_path)
            print("Finished formatting and computing manifest of {0} zip files.".format(len(dataset_paths)))
        else:
            processor.format_and_compute_manifest(args.image_dataset_path, args.is_tagged, args.output_path)

    elif args.command == "compute-features-of-zip":
        processor = ImageDatasetStorageFormatProcessor()

        if dataset_paths:
            for path in tqdm(dataset_paths):
                print("-------------------------------")
                print("Processing {0}...".format(path))
                processor.compute_features_of_zip(path, args.clip_model, args.batch_size)
            print("Finished computing features of {0} zip files.".format(len(dataset_paths)))

        else:
            processor.compute_features_of_zip(args.image_dataset_path, args.clip_model, args.batch_size)

    else:
        # Unreachable
        raise NotImplementedError(
            f"Command {args.command} does not exist.",
        )