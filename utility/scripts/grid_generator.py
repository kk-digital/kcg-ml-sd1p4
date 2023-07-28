import argparse
from PIL import Image
import os
import math
import zipfile
import io
import shutil
import tempfile

def create_image_grid(input_path, n, m, img_size):
    # Determine if the input is a directory or a zip file
    if os.path.isdir(input_path):
        image_dir = input_path
        output_dir = os.path.join(os.path.dirname(input_path), 'grid')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    elif zipfile.is_zipfile(input_path):
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        # assuming the images are located inside the "images" folder within the first folder in the zip file
        first_folder = os.listdir(temp_dir)[0]
        image_dir = os.path.join(temp_dir, first_folder, 'images')
        output_dir = os.path.join(temp_dir, first_folder, 'grid')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        raise ValueError("Input must be a directory or a zip file")

    # List all files in the directory and sort them
    files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])
    print(f'Found {len(files)} files in zip or directory')

    # Grid size
    grid_size = n * m
    grid_count = math.ceil(len(files) / grid_size)

    for grid_index in range(grid_count):
        # Create a new image with the correct size
        img_grid = Image.new('RGB', (n * img_size, m * img_size))

        for i in range(grid_size):
            img_index = grid_index * grid_size + i
            if img_index >= len(files):
                break

            # Open each image, resize it and paste it into the new image
            with Image.open(files[img_index]) as img:
                img_resized = img.resize((img_size, img_size))
                img_grid.paste(img_resized, ((i % n) * img_size, (i // n) * img_size))

        # Save the image grid
        output_file = os.path.join(output_dir, f'grid_{str(grid_index).zfill(4)}.jpg')
        img_grid.save(output_file)
        print(f'Saved image grid to {output_file}')

    # If the input was a zip file, re-zip the directory
    if zipfile.is_zipfile(input_path):
        with zipfile.ZipFile(input_path, 'w') as zipf:
            for folder, _, filenames in os.walk(temp_dir):
                for filename in filenames:
                    absolute_path = os.path.join(folder, filename)
                    relative_path = absolute_path[len(temp_dir) + 1:]  # +1 for the slash
                    zipf.write(absolute_path, arcname=relative_path)
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create an image grid from a directory or a zip of images.')
    parser.add_argument('--input_path', type=str, required=True, help='Input directory or zip file containing images.')
    parser.add_argument('--n', type=int, required=True, help='Number of images in the width of the grid.')
    parser.add_argument('--m', type=int, required=True, help='Number of images in the height of the grid.')
    parser.add_argument('--img_size', type=int, required=True, help='Size of each image in the grid.')
    args = parser.parse_args()

    create_image_grid(args.input_path, args.n, args.m, args.img_size)
