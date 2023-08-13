import cv2
import numpy as np
from PIL import Image

def size_fitness(pil_image):
    """
    This function calculates the 'fitness' score of an object in an image based on its size in relation to the entire image.

    Input:
    pil_image: A PIL.Image object. It should contain the main object of interest on a uniform background.

    Output:
    fitness_score: A floating-point value between 0.0 and 1.0, where 1.0 means the object occupies nearly the entire image,
    and 0.0 implies it has minimal or no presence.

    Function Operations:
    1. Converts the input PIL image to an OpenCV image.
    2. Transforms the image to grayscale and thresholds to identify the object.
    3. Determines the contour (outline) of the main object.
    4. Calculates the ratio of the object's area to the image's total area.
    5. Uses a sigmoid function to transform the raw size score to a value between 0.0 and 1.0.

    """
    # Check if image is None
    if pil_image is None:
        raise ValueError("Image etc is None")
    
    # Check that the image size is 512x512
    if pil_image.size != (512, 512):
        raise ValueError("The image size should be 512x512")

    # Convert the PIL image to an OpenCV image format
    image = np.array(pil_image)[:, :, ::-1].copy()  # Convert RGB to BGR
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to get a binary mask of the object
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assuming the largest contour corresponds to the object of interest
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        _, _, width_b, height_b = cv2.boundingRect(largest_contour)
    else:
        return 0.0

    h, w, c = image.shape
    print('Image_width:  ', w)
    print('Image_height: ', h)
    print('Image_channels:', c)
    
    area_b = width_b * height_b
    area_i = w * h
    
    # Using sigmoid function
    raw_score = 4 - 4 * (area_b / area_i)
    fitness_score = 1 / (1 + np.exp(-raw_score))
    
    assert 0.0 <= fitness_score <= 1.0, "Size fitness value out of bounds!"
    return fitness_score
