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
    
    # Define the central region
    h, w, _ = image.shape
    quarter_h, quarter_w = h // 4, w // 4
    central_region = image[quarter_h:3*quarter_h, quarter_w:3*quarter_w]

    # Create a border mask and extract the border region
    border_mask = np.ones((h, w), dtype=np.uint8)
    border_mask[quarter_h:3*quarter_h, quarter_w:3*quarter_w] = 0
    border_region = cv2.bitwise_and(image, image, mask=border_mask)

    # Adaptive thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    central_thresh = adaptive_thresh[quarter_h:3*quarter_h, quarter_w:3*quarter_w]
    border_thresh = cv2.bitwise_and(adaptive_thresh, adaptive_thresh, mask=border_mask)

    # Compute contour areas
    center_contours, _ = cv2.findContours(central_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    border_contours, _ = cv2.findContours(border_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center_area = sum(cv2.contourArea(cnt) for cnt in center_contours)
    border_area = sum(cv2.contourArea(cnt) for cnt in border_contours)

    # Check color differences
    mean_center_color = np.mean(central_region, axis=(0, 1))
    mean_border_color = np.mean(border_region, axis=(0, 1))
    color_diff = np.linalg.norm(mean_center_color - mean_border_color)

    # Logic for fitness score computation
    # If there's more content in the border region compared to the center, penalize it
    if border_area > center_area:
        return 0.0

    # If the color difference is below a threshold, penalize it
    if color_diff < 20:  # This threshold can be adjusted
        return 0.0

    # Otherwise, base the score on how white the border region is
    fitness_score = 1 - abs(np.mean(mean_border_color) - 255) / 255

    assert 0.0 <= fitness_score <= 1.0, "Background fitness value out of bounds!"
    return fitness_score