import sys
import os
import time

sys.path.insert(0, os.getcwd())
from ga.prompt_generator import *

def test_initialize_prompt_list():
    prompt_list = initialize_prompt_list()
    print(prompt_list)

    assert len(prompt_list) > 0


def test_prompt_generator():
    num_prompts = 1000
    num_phrases = 12
    start_time = time.time()
    generate_prompts(num_prompts, num_phrases)
    print("Time elapsed: {0}s".format(format(time.time() - start_time, ".2f")))