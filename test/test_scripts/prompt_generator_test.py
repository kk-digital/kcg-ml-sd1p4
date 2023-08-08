import sys
import os
sys.path.insert(0, os.getcwd())
from ga.prompt_generator import *

def test_initialize_prompt_list():
    prompt_list = initialize_prompt_list()
    print(prompt_list)

    assert len(prompt_list) > 0


def test_prompt_generator():
    generate_prompts(5, 10)
