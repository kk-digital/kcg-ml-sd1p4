import os
import sys
import zipfile
import numpy as np
from io import BytesIO
import time
import json
from tqdm import tqdm
sys.path.insert(0, os.getcwd())
from ga.prompt_generator import GeneratedPrompt


class PromptListDataset:
    def __init__(self):
        self.zip_ref = None
        self.prompt_paths = []

    # dataset_path - path to the zip file containing the prompt json
    # num_prompts - number of prompts to load, 0 means load all prompts
    def load_prompt_list(self, path_to_zip_file, num_prompts=0):
        start_time = time.time()
        print("Loading prompt list from {}...".format(path_to_zip_file))
        # load zip
        self.zip_ref = zipfile.ZipFile(path_to_zip_file, 'r', compression=zipfile.ZIP_DEFLATED)
        file_paths = self.zip_ref.namelist()

        prompt_count = 0
        for file_path in file_paths:
            # break when number of prompts is achieved
            if num_prompts != 0 and prompt_count >= num_prompts:
                break

            name = os.path.basename(file_path)
            file_extension = os.path.splitext(name)[1]
            if file_extension == ".json":
                self.prompt_paths.append(file_path)
                prompt_count += 1

        print("Prompt list successfully loaded...")
        print("Time elapsed: {0}s".format(format(time.time() - start_time, ".2f")))

    def get_prompt_data(self, index, include_prompt_vector=False):
        # Reading from json file
        with self.zip_ref.open(self.prompt_paths[index]) as f:
            data = f.read().decode("utf-8")
            json_object = json.loads(data)

        prompt_vector = None
        if include_prompt_vector:
            prompt_vector = json_object['prompt-vector']

        prompt_data = GeneratedPrompt(json_object['positive-prompt-str'],
                                      json_object['negative-prompt-str'],
                                      json_object['num-topics'],
                                      json_object['num-modifiers'],
                                      json_object['num-styles'],
                                      json_object['num-constraints'],
                                      prompt_vector)

        return prompt_data
