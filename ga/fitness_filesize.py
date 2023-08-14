from io import BytesIO
from PIL import Image

def filesize_fitness(pil_image, lower_file_size=32*1024, max_file_size=600*1024):
    """
    Compute fitness based on in-memory file size using a linear ramp.
    
    Args:
        pil_image (PIL.Image): The input PIL image.
        lower_file_size (int): The lower bound for the file size.
        max_file_size (int): The upper bound for the file size.
        
    Returns:
        float: The computed fitness score, ranging from 0.00 to 1.00.
    """
    # Check if image is None
    if pil_image is None:
        raise ValueError("Image etc is None")
    
    # Check that the image size is 512x512
    if pil_image.size != (512, 512):
        raise ValueError("The image size should be 512x512")
    
    # Convert PIL image to bytes and get its size
    image_bytes = BytesIO()
    pil_image.save(image_bytes, format="jpeg")
    actual_size = image_bytes.tell()  # Tells the size of the image in bytes

    # If within bounds
    if (lower_file_size <= actual_size) and (actual_size <= max_file_size):
        # Linear ramp formula
        fitness_score = 1 - (actual_size - lower_file_size) / (max_file_size - lower_file_size)
    # If outside bounds
    elif actual_size < lower_file_size:
        fitness_score = 1.0
    else:
        fitness_score = 0.0
    
    # Ensure the fitness score is between 0.00 and 1.00
    assert 0.0 <= fitness_score <= 1.0, "Fitness score out of range"

    return fitness_score
