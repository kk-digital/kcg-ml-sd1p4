
# chad score of a single image using https://github.com/grexzen/SD-Chad/blob/main/simple_inference.py
import torch
import sys
import os

sys.path.insert(0, os.getcwd())
from chad_score.chad_score import get_chad_score
from utils.clip.clip_feature_zip_loader import ClipFeatureZipLoader

def test_chad_scorer():
    # parameters

    img_path = 'test/test_images/clip_segmentation/getty_481292845_77896.jpg'
    model_path = "input/model/chad_score/chad-score-v1.pth"

    batch_size = 1
    clip_model = "ViT-L/14"

    # compute features using clip tools
    loader = ClipFeatureZipLoader()
    loader.load_clip(clip_model)
    feature_vectors = (loader.get_images_feature_vectors(img_path, batch_size))
    feature_vectors = feature_vectors[0]['feature-vector'];
    feature_vectors = torch.tensor(feature_vectors)

    print("Chad score predicted by the model:")
    print(get_chad_score(feature_vectors, model_path))

if __name__ == '__main__':
    test_chad_scorer()