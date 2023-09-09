import sys
import os
import time

sys.path.insert(0, os.getcwd())
from utility.dataset.prompt_list_dataset import PromptListDataset


def test_load_prompt_list_no_limit():
    dataset_path = "./test/test_zip_files/prompt_list_civitai_2_test.zip"
    prompt_dataset = PromptListDataset()

    prompt_dataset.load_prompt_list(dataset_path)

    assert len(prompt_dataset.prompt_paths) == 2


def test_load_prompt_list_with_limit():
    dataset_path = "./test/test_zip_files/prompt_list_civitai_2_test.zip"
    limit = 1
    prompt_dataset = PromptListDataset()

    prompt_dataset.load_prompt_list(dataset_path, limit)

    assert len(prompt_dataset.prompt_paths) == limit