import os
import sys
import random
import clip
import torch
from chad_score.chad_score import ChadScorePredictor
from stable_diffusion.utils_backend import get_device
from stable_diffusion.utils_image import to_pil

DEVICE = get_device()

# Load CLIP
image_features_clip_model, preprocess = clip.load("ViT-L/14", device=DEVICE)

# Load Chad Score
chad_score_model_path = os.path.join('input', 'model', 'chad_score', 'chad-score-v1.pth')
chad_score_predictor = ChadScorePredictor(device=DEVICE)
chad_score_predictor.load_model(chad_score_model_path)

def compute_chad_score_from_pil(pil_image):
    '''
    Compute chad score for a given PIL image.

    Args:
    - pil_image (PIL.Image.Image): Image for which chad score is to be computed.

    Returns:
    - chad_score (torch.Tensor): Chad score for the input image.
    '''
    # Ensure the image is in RGB mode
    assert pil_image.mode == "RGB", "The image should be in RGB mode"
    
    unsqueezed_image = preprocess(pil_image).unsqueeze(0).to(DEVICE)
    
    # Get CLIP encoding of the model
    with torch.no_grad():
        image_features = image_features_clip_model.encode_image(unsqueezed_image)
        raw_chad_score = chad_score_predictor.get_chad_score(image_features.type(torch.cuda.FloatTensor))
    
    #scaled_chad_score = torch.sigmoid(torch.tensor(raw_chad_score)).item()
    
    return raw_chad_score #scaled_chad_score
