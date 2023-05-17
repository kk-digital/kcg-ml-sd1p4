import os
import re
import subprocess
import time
import shutil
from bs4 import BeautifulSoup
import fire

def process_notebooks(input_dir='.', output_dir='../log'):
    # Get current timestamp
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Create log directory with current timestamp
    log_dir = os.path.join(output_dir, current_time)
    os.makedirs(log_dir)

    # Create directory to move un-processed HTML files
    unprocessed_dir = os.path.join(".", "tmp")
    if os.path.exists(unprocessed_dir):
        shutil.rmtree(unprocessed_dir)
    os.makedirs(unprocessed_dir)

    # Find all .ipynb files recursively from input directory
    notebook_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".ipynb"):
                notebook_files.append(os.path.join(root, file))

    # Run jupyter-runner for each notebook file found
    for file in notebook_files:
        cmd = ["jupyter-runner", "--allow-errors", file, "--output-directory", unprocessed_dir, "--debug"]
        print(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Extract errors from HTML files and write them to errors.txt
    errors_file = os.path.join(log_dir, "errors.txt")
    with open(errors_file, "w") as f:
        for root, dirs, files in os.walk(unprocessed_dir, topdown=True):
            for file in files:
                if file.endswith(".html"):
                    with open(os.path.join(root, file), "r") as html_file:
                        f.write("———————————————————————————————————————————————————————————————————————————\n")
                        # Yes, I used a regex to change the .html extension to a .ipynb, probably bad practice
                        # I'm not a coder
                        f.write("Notebook filename: " + re.sub(r"\.html$", ".ipynb", os.path.basename(file)) + "\n")
                        f.write("Notebook path: " + os.path.dirname(os.path.abspath(file)) + "/ \n")
                        f.write("Errors:\n\n")
                        html_content = html_file.read()
                        soup = BeautifulSoup(html_content, 'html.parser')
                        error_matches = soup.find_all("div", attrs={"data-mime-type": "application/vnd.jupyter.stderr"})
                        if error_matches:
                            for i, error in enumerate(error_matches):
                                error_text = error.find("pre").text.strip()
                                f.write(f"Error on cell {i+1}:\n")
                                f.write(f"{error_text}\n\n")
                        else:
                            f.write("No errors\n")
                        f.write("———————————————————————————————————————————————————————————————————————————\n")

    # Move HTML files and errors.txt to logs directory with current timestamp
    for file in os.listdir(unprocessed_dir):
        shutil.move(os.path.join(unprocessed_dir, file), log_dir)

    # Remove un-processed directory
    shutil.rmtree(unprocessed_dir)
    shutil.move(os.path.join(log_dir, "errors.txt"), os.path.join(log_dir, "errors.log"))

if __name__ == '__main__':
    fire.Fire(process_notebooks)
