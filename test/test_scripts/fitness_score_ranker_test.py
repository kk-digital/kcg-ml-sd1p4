import pytest
import os
import sys
sys.path.insert(0, os.getcwd())
import json
from shutil import rmtree
from scripts.fitness_score_ranker import load_fitness_function, rank_images

# Mock fitness function for testing
def mock_fitness_function(image):
    """Mock fitness function that returns 0.5 for all images."""
    return 0.5

def test_rank_images():
    # Define mock arguments
    fitness_function_path = './ga/fitness_bounding_box_centered.py'  
    zip_path = './test/test_zip_files/test-dataset-correct-format.zip'  
    output_path = 'test_output'
    
    # Call the function under test
    fitness_function = load_fitness_function(fitness_function_path)
    rank_images(fitness_function, fitness_function_path, zip_path, output_path)
    
    # Verify that the output directories are created
    assert os.path.exists(output_path)
    for score in range(11):
        assert os.path.exists(os.path.join(output_path, f"{score / 10:.1f}"))
    
    # Verify that the JSON file is created and has content
    json_path = os.path.join(output_path, 'fitness_scores.json')
    assert os.path.exists(json_path)
    
    # Optionally: Verify the content of the JSON file
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        assert json_data["fitness_function_name"] == "fitness_bounding_box_centered"
        assert isinstance(json_data["images"], dict)
        for img, score in json_data["images"].items():
            assert isinstance(img, str)
            assert isinstance(score, (float, int))
    
    # Clean up
    rmtree(output_path)
