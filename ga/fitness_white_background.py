import numpy as np
import cv2
from PIL import Image

def white_background_fitness(pil_image):
    """
    This function calculates the 'fitness' score based on the background color and central color of the image.
    
    Input:
    pil_image: A PIL.Image object of size 512x512.

    Output:
    fitness_score: A floating-point value between 0.0 and 1.0.
        - 1.0 indicates a white background and a discernably different center.
        - 0.0 indicates violations of the desired configuration.
        - Values between 0 and 1 indicate deviations from the ideal.
    """
    # ... [Initial checks remain the same] ...

    # Convert the PIL image to an OpenCV image format
    image = np.array(pil_image)[:, :, ::-1].copy()  # Convert RGB to BGR

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the central region
    h, w = gray.shape
    quarter_h, quarter_w = h // 4, w // 4
    central_region = gray[quarter_h:3*quarter_h, quarter_w:3*quarter_w]

    # Mask the central region out to compute the mean intensity of the border region
    border_mask = np.ones_like(gray)
    border_mask[quarter_h:3*quarter_h, quarter_w:3*quarter_w] = 0
    border_pixels = gray[border_mask == 1]
    
    # Compute the mean pixel intensity of the border
    mean_border_intensity = np.mean(border_pixels)
    mean_center_intensity = np.mean(central_region)
    
    # Strict condition for white border
    WHITE_THRESHOLD = 100  # Adjust as needed. If the border's mean intensity deviates more than this value from pure white, it's considered not white.
    if abs(mean_border_intensity - 255) > WHITE_THRESHOLD:
        return 0.0

    # Ensure the central region is not too bright compared to the border
    if mean_center_intensity > (mean_border_intensity - 10):  #function becomes stricter try 1:
        return 0.0

    deviation = abs(mean_border_intensity - 255) / 255
    fitness_score = 1 - deviation**2

    assert 0.0 <= fitness_score <= 1.0, "Background fitness value out of bounds!"
    return fitness_score
