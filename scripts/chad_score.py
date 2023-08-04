import argparse
import os
import sys

import torch

sys.path.insert(0, os.getcwd())

from stable_diffusion.utils_backend import get_device
from chad_score.chad_score import ChadScorePredictor
from utility.utils.clip.clip_feature_zip_loader import ClipFeatureZipLoader


def parse_arguments():
    """Command-line arguments for 'classify' command."""
    parser = argparse.ArgumentParser(description="Chad Score.")

    # Arguments for 'classify'
    parser.add_argument('--model-path', type=str,
                        help='Path to the model used for classifying the items in the dataset')
    parser.add_argument('--image-path', type=str, help='Path to image to score')

    return parser.parse_args()


def main():
    args = parse_arguments()
    img_path = args.image_path
    model_path = args.model_path
    device = get_device()

    batch_size = 1
    clip_model = "ViT-L/14"

    # compute features using clip tools
    loader = ClipFeatureZipLoader()
    loader.load_clip(clip_model)
    feature_vectors = (loader.get_images_feature_vectors(img_path, batch_size))[0]['feature-vector'];
    feature_vectors = torch.tensor(feature_vectors, device=device)

    chad_score_predictor = ChadScorePredictor(device=device)
    chad_score_predictor.load_model()
    print("Chad score predicted by the model:")
    print(chad_score_predictor.get_chad_score(feature_vectors))


if __name__ == '__main__':
    main()
