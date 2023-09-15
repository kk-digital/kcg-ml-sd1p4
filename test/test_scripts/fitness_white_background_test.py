import pytest
import os
import sys
sys.path.insert(0, os.getcwd())
from PIL import Image
import numpy as np
import cv2
from ga.fitness_white_background import white_background_fitness

def test_centered_fitness():
    pil_image = Image.open("test/test_images/test_img.jpg")
    fitness_score = white_background_fitness(pil_image)
    
    assert 0.0 <= fitness_score <= 1.0, f"Fitness score out of bounds"