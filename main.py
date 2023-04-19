import os
import subprocess

# path to text-to-image script
text_to_image_script = "replaceme"

# input and output directories
input_dir = "/input"
output_dir = "/output"

# loop through all text files in input directory
for file in os.listdir(input_dir):
    if file.endswith(".txt"):
        # construct input and output paths
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, os.path.splitext(file)[0] + ".png")

        # call text-to-image script with input and output paths as arguments
        subprocess.call(["python", text_to_image_script, input_path, output_path])
