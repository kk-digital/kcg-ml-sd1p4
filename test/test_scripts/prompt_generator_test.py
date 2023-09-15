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


# def test_prompt_proportional_sampling():
#     csv_dataset_path = "./input/civit_ai_data_phrase_count_v6.csv"
#     csv_phrase_limit = 50
#     phrases, \
#         phrases_token_size, \
#         positive_count_list, \
#         negative_count_list = initialize_prompt_list_from_csv(csv_dataset_path, csv_phrase_limit)
#
#     sorted_phrases, \
#         sorted_token_size, \
#         sorted_count, \
#         sorted_cumulative_sum = get_sorted_list_with_cumulative(phrases, phrases_token_size, positive_count_list)
#
#     print("\nphrase, token size, count, cumulative_sum")
#     for i in range(len(sorted_phrases)):
#         print("{}, {}, {}, {}".format(sorted_phrases[i].Phrase, sorted_token_size[i], sorted_count[i], sorted_cumulative_sum[i]))
#
#     index = find_first_element_binary_search(sorted_cumulative_sum,  441318)
#     print(index)
