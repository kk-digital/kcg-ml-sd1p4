import cv2
import numpy as np

def size_fitness(image_path):
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        raise ValueError(f"Error loading image at {image_path}")

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

    height_i, width_i = image.shape[:2]
    
    area_b = width_b * height_b
    area_i = width_i * height_i
    
    # Using sigmoid function
    raw_score = 4 - 4 * (area_b / area_i)
    f_size = 1 / (1 + np.exp(-raw_score))
    
    assert 0.0 <= f_size <= 1.0, "Size fitness value out of bounds!"
    return f_size
