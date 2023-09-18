# chad score of a single image using https://github.com/grexzen/SD-Chad/blob/main/simple_inference.py
import os
import sys

import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

sys.path.insert(0, os.getcwd())
from model.util_clip import ClipModelHuggingface
from chad_score.chad_score import ChadScorePredictor


def test_clip_vector():
    # parameters
    device = 'cuda:0'

    # Load the clip model
    util_clip = ClipModelHuggingface(device=device)
    util_clip.load_model()

    # Create a random tensor with dimensions (1, 3, 512, 512)
    random_tensor = torch.rand(1, 3, 512, 512, device=device)

    images = torch.clamp((random_tensor + 1.0) / 2.0, min=0.0, max=1.0)

    images_cpu = images.permute(0, 2, 3, 1)
    images_cpu = images_cpu.detach().cpu().float().numpy()

    image_list = []
    # Save images
    for i, img in enumerate(images_cpu):
        img = Image.fromarray((255. * img).astype(np.uint8))
        image_list.append(img)

    image = image_list[0]

    image_features_a = util_clip.get_image_features(image)
    image_features_a.to(torch.float32)

    ## Normalize the image tensor
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(-1, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(-1, 1, 1)

    normalized_image_tensor = (images - mean) / std

    # Resize the image to [N, C, 224, 224]
    transform = transforms.Compose([transforms.Resize((224, 224))])
    resized_image_tensor = transform(normalized_image_tensor)

    print(resized_image_tensor.shape)

    # Get the CLIP features
    image_features_b = util_clip.model.encode_image(resized_image_tensor)
    image_features_b = image_features_b.squeeze(0)

    image_features_b = image_features_b.to(torch.float32)

    mse = torch.mean((image_features_a - image_features_b) ** 2)

    print("MSE : ", mse.item())

    # Load default chad model
    # hard coded for now
    chad_score_model_path = "input/model/chad_score/chad-score-v1.pth"
    chad_score_model_name = os.path.basename(chad_score_model_path)
    chad_score_predictor = ChadScorePredictor(device=device)
    chad_score_predictor.load_model(chad_score_model_path)

    chad_score_a = chad_score_predictor.get_chad_score(image_features_a);
    chad_score_b = chad_score_predictor.get_chad_score(image_features_b);

    print("Chad Score Method A : ", chad_score_a)
    print("Chad Score Method B : ", chad_score_b)

if __name__ == '__main__':
    test_clip_vector()
