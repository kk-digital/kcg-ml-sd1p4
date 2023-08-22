import numpy as np
from PIL import Image

def fitness_pixel_value(pil_image):
    """
    This function calculates a fitness score for an image based on the number of pixels with values less than 32 in the center of the image.
    
    Input:
    pil_image: A PIL.Image object of size 512x512.

    Output:
    fitness_score: A floating-point value between 0.0 and 1.0.
        - 1.0 indicates the center 1/4 of the image has pixels with values less than 32.
        - Values between 0 and 1 indicate deviations from this ideal.
    """

    # Ensure the image is of size 512x512
    if pil_image.size != (512, 512):
        raise ValueError("The image size should be 512x512")

    # Convert the image to grayscale for intensity calculations
    gray_image = pil_image.convert('L')
    np_image = np.array(gray_image)

    # Define the center region, which should be 256x256 (half of 512x512)
    h, w = np_image.shape
    quarter_h, quarter_w = h // 4, w // 4
    center_region = np_image[quarter_h:3*quarter_h, quarter_w:3*quarter_w]

    # Count pixels with value less than 32 in the center
    count_below_32 = np.sum(center_region < 32)

    # Ideal count is the total number of pixels in the center region 
    ideal_count = (h // 2) * (w // 2)

    # Fitness score is the ratio of the count_below_32 to the ideal count
    fitness_score = count_below_32 / ideal_count

    # Calculate the deviation from the ideal fitness score
    deviation = 1 - fitness_score

    # Calculate fitness score using the formula
    fitness_score = 1 - deviation**2

    assert 0.0 <= fitness_score <= 1.0, "Background fitness value out of bounds!"

    return fitness_score

