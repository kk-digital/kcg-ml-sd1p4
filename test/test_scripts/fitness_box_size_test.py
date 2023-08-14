import pytest
import os
import sys
sys.path.insert(0, os.getcwd())
from PIL import Image
from ga.fitness_bounding_box_size import size_fitness  # Assuming you save the function in ga/fitness_size.py

def test_size_fitness():
    pil_image = Image.open("test/test_images/test_img.jpg")  # Make sure to put the appropriate test image here
    fitness_score = size_fitness(pil_image)

    assert 0.0 <= fitness_score <= 1.0, f"Size fitness score out of bounds"
