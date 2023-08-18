import numpy as np
import cv2
from PIL import Image

def white_background_fitness(pil_image):
    """
    This function calculates the 'fitness' score based on the background color of the image.
    
    Input:
    pil_image: A PIL.Image object of size 512x512.

    Output:
    fitness_score: A floating-point value between 0.0 and 1.0.
        - 1.0 indicates a white background.
        - 0.0 indicates a black background.
        - Values between 0 and 1 indicate varying shades of gray.

    Function Operations:
    1. Converts the input PIL image to an OpenCV image.
    2. Transforms the image to grayscale.
    3. Computes the mean pixel intensity of the grayscale image.
    4. Normalizes the mean value to obtain the fitness score.

    """
    # Check if image is None
    if pil_image is None:
        raise ValueError("Image etc is None")

    # Check that the image size is 512x512
    if pil_image.size != (512, 512):
        raise ValueError("The image size should be 512x512")
    
    # Check if image mode is not RGB
    if pil_image.mode != "RGB":
        raise ValueError("The provided image is not in RGB mode. Please provide an RGB image.")

    # Convert the PIL image to an OpenCV image format
    image = np.array(pil_image)[:, :, ::-1].copy()  # Convert RGB to BGR

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute the mean pixel intensity
    mean_intensity = np.mean(gray)

    # Normalize the mean pixel intensity to get the fitness score
    fitness_score = mean_intensity / 255.0

    assert 0.0 <= fitness_score <= 1.0, "Background fitness value out of bounds!"
    return fitness_score
