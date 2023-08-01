import argparse
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import math
import zipfile
import io
import shutil
import tempfile

def create_image_grid(input_path, output_path, rows, columns, img_size):
    # If output directory does not exist, create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Determine if the input is a directory or a zip file
    if os.path.isdir(input_path):
        root_dir = input_path
    elif zipfile.is_zipfile(input_path):
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        root_dir = temp_dir
    else:
        raise ValueError("Input must be a directory or a zip file")

    # List all files in the directory and sort them
    files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in [f for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]:
            files.append(os.path.join(dirpath, filename))
    files.sort(key=lambda f: os.path.basename(f))

    print(f'Found {len(files)} image files in zip or directory')

    # Grid size
    grid_size = rows * columns
    grid_count = math.ceil(len(files) / grid_size)

    for grid_index in range(grid_count):
        # Create a new image with the correct size
        img_grid = Image.new('RGB', (columns * img_size, rows * img_size))

        for i in range(grid_size):
            img_index = grid_index * grid_size + i
            if img_index >= len(files):
                break

            # Open each image, resize it and paste it into the new image
            with Image.open(files[img_index]) as img:
                img_resized = img.resize((img_size, img_size))
                img_grid.paste(img_resized, ((i % columns) * img_size, (i // columns) * img_size))

        # Save the image grid
        output_file = os.path.join(output_path, f'grid_{str(grid_index).zfill(4)}.jpg')
        img_grid.save(output_file)
        print(f'Saved image grid to {output_file}')

    # If the input was a zip file, delete the temporary directory
    if zipfile.is_zipfile(input_path):
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create an image grid from a directory or a zip of images.')
    parser.add_argument('--input_path', type=str, required=True, help='Input directory or zip file containing images.')
    parser.add_argument('--output_path', type=str, required=True, help='Output directory for grid images.')
    parser.add_argument('--rows', type=int, required=True, help='Number of images in the height of the grid.')
    parser.add_argument('--columns', type=int, required=True, help='Number of images in the width of the grid.')
    parser.add_argument('--img_size', type=int, required=True, help='Size of each image in the grid.')
    args = parser.parse_args()

    create_image_grid(args.input_path, args.output_path, args.rows, args.columns, args.img_size)
