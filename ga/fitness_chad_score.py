import os
import sys
import torch
from chad_score.chad_score import ChadScorePredictor
from stable_diffusion.utils_backend import get_device

DEVICE = get_device()


# Load Chad Score
chad_score_model_path = os.path.join('input', 'model', 'chad_score', 'chad-score-v1.pth')
chad_score_predictor = ChadScorePredictor(device=DEVICE)
chad_score_predictor.load_model(chad_score_model_path)


def compute_chad_score_from_features(feature_vector):
    '''
    Compute chad score for a given feature vector.

    Args:
    - feature_vector (torch.Tensor): Feature vector for which chad score is to be computed.

    Returns:
    - chad_score (torch.Tensor): Chad score for the input feature vector.
    '''
    
    # Ensure the feature_vector is a torch.Tensor
    assert isinstance(feature_vector, torch.Tensor), "The feature vector should be a torch.Tensor"

    with torch.no_grad():
        # Get the chad score directly using the feature vector
        raw_chad_score = chad_score_predictor.get_chad_score(feature_vector.type(torch.cuda.FloatTensor))
    
    scaled_chad_score = torch.sigmoid(torch.tensor(raw_chad_score)).item()
    
    return raw_chad_score, scaled_chad_score

