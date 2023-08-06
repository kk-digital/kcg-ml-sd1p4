import cv2
import numpy as np
from PIL import Image

class NumberBoundingBox:
    def __init__(self, pil_image):
        self.image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def get_bounding_boxes(self):
        # Convert the image to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract bounding boxes
        bounding_boxes = [cv2.boundingRect(c) for c in contours]

        return bounding_boxes

    def draw_bounding_boxes(self):
        img_with_boxes = self.image.copy()
        for (x, y, w, h) in self.get_bounding_boxes():
            cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return Image.fromarray(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))


