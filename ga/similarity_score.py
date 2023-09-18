
import os
import sys
import torch

base_dir = os.getcwd()
sys.path.insert(0, base_dir)


def get_similarity_score(image_features, target_features):

    image_features_magnitude = torch.norm(image_features)
    target_features_magnitude = torch.norm(target_features)

    image_features = image_features / image_features_magnitude
    target_features = target_features / target_features_magnitude

    image_features = image_features.squeeze(0)

    similarity = torch.dot(image_features, target_features)

    fitness = similarity

    return fitness

