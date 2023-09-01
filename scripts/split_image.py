from PIL import Image
import argparse

def parse_arguments():
    """Command-line arguments for 'classify' command."""
    parser = argparse.ArgumentParser(
        description="Chad sort, takes in an image database and sorts it by chad score into many folders.")

    parser.add_argument('--image_path', type=str, help='Path to the image to use')
    parser.add_argument('--image_width', type=int, default=512,)
    parser.add_argument('--image_height', type=int, default=512)
    parser.add_argument('--output_image_width', type=int, default=64)
    parser.add_argument('--output_image_height', type=int, default=64)

    return parser.parse_args()

def split_image_into_subimages(input_image_path, output_path, subimage_size):
    # Open the input image
    input_image = Image.open(input_image_path)

    # Get the dimensions of the input image
    width, height = input_image.size

    # Define the size of the sub-images (64x64 in this case)
    sub_width, sub_height = subimage_size

    # Calculate the number of sub-images in both dimensions
    num_subimages_width = width // sub_width
    num_subimages_height = height // sub_height

    # Iterate through the sub-images
    for i in range(num_subimages_width):
        for j in range(num_subimages_height):
            # Calculate the coordinates of the top-left corner of the current sub-image
            left = i * sub_width
            upper = j * sub_height
            right = left + sub_width
            lower = upper + sub_height

            # Crop the sub-image
            sub_image = input_image.crop((left, upper, right, lower))

            # Save the sub-image
            sub_image.save(f"{output_path}/subimage_{i}_{j}.png")

if __name__ == "__main__":
    input_image_path = "input_image.jpg"  # Replace with your input image path
    output_path = "output_subimages"      # Output directory for sub-images
    subimage_size = (64, 64)              # Size of each sub-image

    # Create the output directory if it doesn't exist
    import os
    os.makedirs(output_path, exist_ok=True)

    # Split the input image into sub-images
    split_image_into_subimages(input_image_path, output_path, subimage_size)