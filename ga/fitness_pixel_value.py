import numpy as np
from PIL import Image

def fitness_pixel_value(pil_image):
    """
    This function calculates a fitness score for an image based on the number of pixels with values less than 80 in the center of the image.
    
    Input:
    pil_image: A PIL.Image object of size 512x512.

    Output:
    fitness_score: A floating-point value between 0.0 and 1.0.
        - 1.0 indicates the center 1/4 of the image has pixels with values less than 80.
        - Values between 0 and 1 indicate deviations from this ideal.
    """

    # Ensure the image is of size 512x512
    if pil_image.size != (512, 512):
        raise ValueError("The image size should be 512x512")

    # Convert the image to grayscale for intensity calculations
    gray_image = pil_image.convert('L')
    np_image = np.array(gray_image)

    # Define the center region, which should be 256x256 (half of 512x512)
    center_start = 512 // 4
    center_end = 3 * 512 // 4
    center_region = np_image[center_start:center_end, center_start:center_end]

    # Count pixels with value less than 16 in the center
    count_below_80 = np.sum(center_region < 16)

    # Ideal count is the total number of pixels in the center region (256x256)
    ideal_count = (512 // 2) * (512 // 2)

    # Fitness score is the ratio of the count_below_80 to the ideal count
    fitness_score = count_below_80 / ideal_count

    # Clamp the value between 0 and 1
    fitness_score = max(0.0, min(1.0, fitness_score))

    return fitness_score
