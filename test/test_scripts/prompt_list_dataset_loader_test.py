import sys
import os
import time

sys.path.insert(0, os.getcwd())
from utility.dataset.prompt_list_dataset import PromptListDataset


def test_load_prompt_list_no_limit():
    dataset_path = "./test/test_zip_files/prompt_list_civitai_2_test.zip"
    prompt_dataset = PromptListDataset()
    index_to_check = 0
    expected_pos_prompt_str = "lips, raw photo, best quality, medium breasts, small breasts, dynamic pose, " \
                              "spread legs, abs, high quality, perfect eyes, perfect anatomy, 8k, looking at " \
                              "viewer, (masterpiece:1.2), milf, detailed eyes, intricate, bangs, detailed face, " \
                              "masterpiece, beautiful eyes, close-up"
    expected_num_of_prompts = 2

    prompt_dataset.load_prompt_list(dataset_path)
    prompt_data = prompt_dataset.get_prompt_data(index_to_check)

    assert prompt_data.positive_prompt_str == expected_pos_prompt_str
    assert len(prompt_dataset.prompt_paths) == expected_num_of_prompts


def test_load_prompt_list_with_limit():
    dataset_path = "./test/test_zip_files/prompt_list_civitai_2_test.zip"
    limit = 1
    prompt_dataset = PromptListDataset()

    prompt_dataset.load_prompt_list(dataset_path, limit)

    assert len(prompt_dataset.prompt_paths) == limit