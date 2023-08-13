import cv2
import numpy as np

def centered_fitness(image_path):
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
        x_b, y_b, width_b, height_b = cv2.boundingRect(largest_contour)
    else:
        return 0.0

    height_i, width_i = image.shape[:2]
    x_center_b = x_b + width_b / 2
    y_center_b = y_b + height_b / 2
    x_i, y_i = width_i / 2, height_i / 2
    
    distance = np.sqrt((x_center_b - x_i)**2 + (y_center_b - y_i)**2)
    max_distance = np.sqrt((0 - x_i)**2 + (0 - y_i)**2)
    
    # Using sigmoid function
    raw_score = 1 - (distance / max_distance)
    fitness_score = 1 / (1 + np.exp(-raw_score))
    
    assert 0.0 <= fitness_score <= 1.0, "Centered fitness value out of bounds!"
    return fitness_score





