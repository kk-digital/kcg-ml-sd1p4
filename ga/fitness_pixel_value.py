from PIL import Image
import numpy as np

def fitness_pixel_value(pil_image):
    """
    This function calculates a fitness score for an image based on the number of pixels with values less than 80.
    
    Input:
    pil_image: A PIL.Image object of size 512x512.

    Output:
    fitness_score: A floating-point value between 0.0 and 1.0.
        - 1.0 indicates exactly 1/4 of the image's pixels have values less than 80.
        - Values between 0 and 1 indicate deviations from this ideal.
    """

    # Ensure the image is of size 512x512
    if pil_image.size != (512, 512):
        raise ValueError("The image size should be 512x512")

    # Convert the image to grayscale for intensity calculations
    gray_image = pil_image.convert('L')
    np_image = np.array(gray_image)

    # Count pixels with value less than 80
    count_below_80 = np.sum(np_image < 80)

    # Ideal count is 1/4 of the total number of pixels in the image
    ideal_count = (512 * 512) // 4

    # Fitness score is the ratio of the count_below_80 to the ideal count
    fitness_score = count_below_80 / ideal_count

    # Clamp the value between 0 and 1
    fitness_score = max(0.0, min(1.0, fitness_score))

    return fitness_score
