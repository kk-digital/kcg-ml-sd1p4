#model_chad_score.py

import torch

from chad_score.chad_score import ChadScorePredictor
#import clip

#TODO: Using wrong device function
#TODO: Loading wrong/different clip model? Or using correct library
#TODO: make class and store the clip model.
#TODO: make save/load function
#TODO: model should not have own copy of clip encoder, which should be global

#todo, wrong device function

def LoadChadScoreModel(ModelPath):
	device = torch.device('cuda:0')
	chad_score_predictor = ChadScorePredictor(device=device)
	chad_score_predictor.load_model(ModelPath)
	return chad_score_predictor