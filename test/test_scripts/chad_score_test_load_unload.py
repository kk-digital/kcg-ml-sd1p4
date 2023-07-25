import os
import sys

sys.path.insert(0, os.getcwd())
from chad_score.chad_score_predictor import ChadPredictor








def test_chad_load_unload():
    model_path = './chad_score/models/chad-score-v1.pth'

    chad_predictor = ChadPredictor()

    chad_predictor.load(model_path)
    chad_predictor.unload()


if __name__ == '__main__':
    test_chad_load_unload()