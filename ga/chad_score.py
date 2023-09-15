
import os
import sys
import torch

base_dir = os.getcwd()
sys.path.insert(0, base_dir)


def get_chad_score(chad_score_predictor, image_features):

    chad_score = chad_score_predictor.get_chad_score_tensor(image_features)

    fitness = chad_score

    return fitness

