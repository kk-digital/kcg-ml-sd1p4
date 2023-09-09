import os
import sys
import zipfile
import numpy as np
from io import BytesIO
import time
from tqdm import tqdm
sys.path.insert(0, os.getcwd())
from ga.prompt_generator import GeneratedPrompt

class PromptListDataset:
    def __init__(self):
        self.prompt_list = []

    # dataset_path - path to the zip file containing the prompt npzs
    # num_prompts - number of prompts to load, 0 means load all prompts
    def load_prompt_list(self, path_to_zip_file, num_prompts=0):
        start_time = time.time()
        print("Loading prompt list from {}...".format(path_to_zip_file))
        # load zip
        zip_ref = zipfile.ZipFile(path_to_zip_file, 'r', compression=zipfile.ZIP_DEFLATED)
        file_paths = zip_ref.namelist()

        prompt_count = 0
        for file_path in tqdm(file_paths):
            # break when number of prompts is achieved
            if num_prompts != 0 and prompt_count >= num_prompts:
                break

            name = os.path.basename(file_path)
            file_extension = os.path.splitext(name)[1]
            if file_extension == ".npz":
                data = zip_ref.read(file_path)

                # load npz data
                data = np.load(BytesIO(data), allow_pickle=True)
                npz_data = data['data'].tolist()
                prompt_data = GeneratedPrompt(npz_data['positive-prompt-str'],
                                              npz_data['negative-prompt-str'],
                                              npz_data['num-topics'],
                                              npz_data['num-modifiers'],
                                              npz_data['num-styles'],
                                              npz_data['num-constraints'],
                                              npz_data['prompt-vector'],
                                              npz_data['positive-prompt-embedding'],
                                              npz_data['negative-prompt-embedding'])
                # add to prompt list
                self.prompt_list.append(prompt_data)

                prompt_count += 1

        print("Prompt list successfully loaded...")
        print("Time elapsed: {0}s".format(format(time.time() - start_time, ".2f")))
