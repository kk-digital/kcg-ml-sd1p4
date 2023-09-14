# prompt_generator.py
import random
import tiktoken
import sys
import os
import shutil
import json
import math
import csv
import torch
from tqdm import tqdm
import numpy as np

base_directory = "./"
sys.path.insert(0, base_directory)


class GeneratedPrompt:
    def __init__(self, positive_prompt_str: str, negative_prompt_str: str, num_topics: int, num_modifiers: int,
                 num_styles: int, num_constraints: int, prompt_vector: []):
        self.positive_prompt_str = positive_prompt_str
        self.negative_prompt_str = negative_prompt_str
        self.num_topics = num_topics
        self.num_modifiers = num_modifiers
        self.num_styles = num_styles
        self.num_constraints = num_constraints

        # prompt_vector is a vector of -1, 0, or 1
        # 1 - used phrase for positive prompt
        # 0 - unused phrase
        # -1 - used for negative prompt
        self.prompt_vector = prompt_vector

    def get_positive_prompt_str(self):
        return self.positive_prompt_str

    def get_negative_prompt_str(self):
        return self.negative_prompt_str

    def to_json(self):
        return {'positive-prompt-str': self.positive_prompt_str,
                'negative-prompt-str': self.negative_prompt_str,
                'prompt-vector': self.prompt_vector,
                'num-topics': self.num_topics,
                'num-modifiers': self.num_modifiers,
                'num-styles': self.num_styles,
                'num-constraints': self.num_constraints,
                }


class PromptData:
    def __init__(self, index: int, phrase: str):
        self.Index = index

        # type is a list since
        # a phrase can have multiple
        # types. Like "chibi" can be
        # a topic and also a style.
        #
        # types can be:
        # topic - ex. "waifu"
        # modifier - ex. "beautiful"
        # style - ex. "water color"
        # constraint - ex. "white background"
        self.Types = []
        self.Phrase = phrase


class PromptList():
    def __init__(self):
        self.Prompts = []

    def is_phrase_exist(self, phrase: str):
        for prompt in self.Prompts:
            if prompt.Phrase == phrase:
                return True

        return False

    def add_phrase(self, phrase: str):
        if not self.is_phrase_exist(phrase):
            index = len(self.Prompts)
            new_prompt = PromptData(index, phrase)
            self.Prompts.append(new_prompt)
        else:
            print("Phrase: {} already exists".format(phrase))

    def add_phrases(self, phrases: []):
        for phrase in phrases:
            self.add_phrase(phrase)

    def add_type_to_phrase(self, phrase: str, prompt_type: str):
        # check if phrase exist
        prompt_data = [prompt for prompt in self.Prompts if (prompt.Phrase == phrase)]

        # if exist add type to phrase
        if len(prompt_data) != 0:
            prompt_data = prompt_data[0]
            # check first if type is already in list
            is_prompt_type_exists = len(
                [prompt_type for prompt_type in prompt_data.Types if prompt_type == prompt_type]) > 0
            if not is_prompt_type_exists:
                prompt_data.Types.append(prompt_type)
            else:
                raise Exception("Trying to add existing type:{0} to phrase:{1}".format(prompt_type, phrase))
        # if not, make phrase and add type
        else:
            self.add_phrase(phrase)
            self.add_type_to_phrase(phrase, prompt_type)

    def add_types_to_phrase(self, phrase: str, types: []):
        for prompt_type in types:
            self.add_type_to_phrase(phrase, prompt_type)

    def add_topic_phrases(self, phrases: []):
        for phrase in phrases:
            self.add_phrase(phrase)
            self.add_type_to_phrase(phrase, prompt_type="topic")

    def add_style_phrases(self, phrases: []):
        for phrase in phrases:
            self.add_phrase(phrase)
            self.add_type_to_phrase(phrase, prompt_type="style")

    def add_modifier_phrases(self, phrases: []):
        for phrase in phrases:
            self.add_phrase(phrase)
            self.add_type_to_phrase(phrase, prompt_type="modifier")

    def add_constraint_phrases(self, phrases: []):
        for phrase in phrases:
            self.add_phrase(phrase)
            self.add_type_to_phrase(phrase, prompt_type="constraint")


def initialize_prompt_list():
    prompt_list = PromptList()

    prompt_list.add_topic_phrases(["chibi", "waifu", "cyborg", "dragon", "android", "mecha", "companion", "furry",
                                   "robot", "mercenary", "wizard",
                                   "pet", "shapeshifter", "pilot", "time traveler", "engineer", "slaver",
                                   "samurai"])

    prompt_list.add_modifier_phrases(["beautiful", "unreal", "masterpiece", "gorgeous", "stunning", "captivating",
                                      "breathtaking", "exquisite", "magnificent", "majestic", "elegant", "sublime",
                                      "futuristic", "cyberpunk", "hi-tech", "advanced", "innovative", "modern",
                                      "fantasy", "mythical", "scifi", "character", "full character", "black and white",
                                      "colored", "video game"])

    prompt_list.add_constraint_phrases(["white background", "no background", "not centered", "centered"])

    prompt_list.add_style_phrases(
        ["sketch", "line drawing", "side scrolling", "Highly stylized digital artwork of (an ocean of many hearts)",
         "trending on artstation", "incredible vibrant colors", "dynamic epic composition", "foamy stylized water",
         "ray tracing", "traditional art by studio ghibli"])

    return prompt_list.Prompts


def initialize_prompt_list_from_csv(csv_dataset_path, csv_phrase_limit=0):
    prompt_list = PromptList()
    phrase_token_size_list = []
    positive_count_list = []
    negative_count_list = []
    with open(csv_dataset_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                # index,total count,positive count,negative count,token size,phrase str
                phrase = row[5]

                index = len(prompt_list.Prompts)
                new_prompt = PromptData(index, phrase)
                new_prompt.Types.append("topic")
                prompt_list.Prompts.append(new_prompt)

                # add token count
                phrase_token_size = int(row[4])
                phrase_token_size_list.append(phrase_token_size)

                # add positive count
                positive_count = int(row[2])
                positive_count_list.append(positive_count)

                # add negative count
                negative_count = int(row[3])
                negative_count_list.append(negative_count)

                line_count += 1

            if csv_phrase_limit != 0 and line_count > csv_phrase_limit:
                break

    return prompt_list.Prompts, phrase_token_size_list, positive_count_list, negative_count_list


# Function to generate prompts
# To be deprecated
# This only generates positive prompt
# Still used in GA scripts
def generate_prompts(prompt_count, prompt_phrase_length):
    max_token_size = 75
    prompts = initialize_prompt_list()
    prompt_list = []
    enc = tiktoken.get_encoding("cl100k_base")

    len_prompt_phrases = len(prompts)
    for i in range(0, prompt_count):
        num_tokens = 100
        while num_tokens > max_token_size:
            positive_prompt = []
            prompt_vector = [0] * len(prompts)
            for j in range(0, prompt_phrase_length):
                while True:
                    random_index = random.randint(0, len_prompt_phrases-1)
                    if prompt_vector[random_index] == 0:
                        prompt_index = random_index
                        random_prompt = prompts[prompt_index]
                        break

                # update used array
                prompt_vector[prompt_index] = 1
                positive_prompt.append(random_prompt)

            positive_prompt_str = ', '.join([prompt.Phrase for prompt in positive_prompt])

            # check the length of prompt embedding.
            # if it's more than 75, then regenerate/reroll
            # (max token size is 75)
            prompt_tokens = enc.encode(positive_prompt_str)
            num_tokens = len(prompt_tokens)

        num_topics = len([prompt.Phrase for prompt in positive_prompt if "topic" in prompt.Types])
        num_modifiers = len([prompt.Phrase for prompt in positive_prompt if "modifier" in prompt.Types])
        num_styles = len([prompt.Phrase for prompt in positive_prompt if "style" in prompt.Types])
        num_constraints = len([prompt.Phrase for prompt in positive_prompt if "constraint" in prompt.Types])

        prompt_list.append(
            GeneratedPrompt(positive_prompt_str, "", num_topics, num_modifiers, num_styles,
                            num_constraints, prompt_vector))

    return prompt_list


def generate_prompts_from_csv(csv_dataset_path,
                              csv_phrase_limit,
                              prompt_count,
                              positive_prefix="",
                              dataset_output="",
                              positive_ratio_threshold=3,
                              negative_ratio_threshold=3,
                              use_threshold=True):
    # max token size
    max_token_size = 75
    comma_token_size = 1

    phrases, \
        phrases_token_size,\
        positive_count_list,\
        negative_count_list = initialize_prompt_list_from_csv(csv_dataset_path, csv_phrase_limit)

    positive_prefix_token_size = 0
    if positive_prefix != "":
        # get token size for prefix
        enc = tiktoken.get_encoding("cl100k_base")
        positive_prefix_prompt_tokens = enc.encode(positive_prefix)
        positive_prefix_token_size = len(positive_prefix_prompt_tokens)

    print("Generating {} prompts...".format(prompt_count))
    count = 0
    for i in tqdm(range(0, prompt_count)):
        positive_prompt_total_token_size = positive_prefix_token_size
        negative_prompt_total_token_size = 0
        positive_prompt = []
        negative_prompt = []
        prompt_vector = [0] * len(phrases)

        # positive prompt
        while positive_prompt_total_token_size < max_token_size:
            random_prompt = random.choice(
                [item for item in phrases if (prompt_vector[item.Index] == 0)])
            prompt_index = random_prompt.Index

            # count ratio
            if use_threshold is True and negative_count_list[prompt_index] != 0:
                chosen_phrase_positive_ratio = positive_count_list[prompt_index]/negative_count_list[prompt_index]
                if chosen_phrase_positive_ratio < positive_ratio_threshold:
                    # then dont use this phrase
                    continue

            chosen_phrase_size = phrases_token_size[prompt_index]
            sum_token_size = positive_prompt_total_token_size + chosen_phrase_size + comma_token_size
            if sum_token_size < max_token_size:
                # update used array
                prompt_vector[prompt_index] = 1
                positive_prompt.append(random_prompt)
                positive_prompt_total_token_size = sum_token_size
            else:
                break

        # negative prompt
        while negative_prompt_total_token_size < max_token_size:
            random_prompt = random.choice(
                [item for item in phrases if (prompt_vector[item.Index] == 0)])
            prompt_index = random_prompt.Index

            # count ratio
            if use_threshold is True and positive_count_list[prompt_index] != 0:
                chosen_phrase_negative_ratio = negative_count_list[prompt_index] / positive_count_list[prompt_index]
                if chosen_phrase_negative_ratio < negative_ratio_threshold:
                    # then dont use this phrase
                    continue

            chosen_phrase_size = phrases_token_size[prompt_index]
            sum_token_size = negative_prompt_total_token_size + chosen_phrase_size + comma_token_size
            if sum_token_size < max_token_size:
                # update used array
                prompt_vector[prompt_index] = -1
                negative_prompt.append(random_prompt)
                negative_prompt_total_token_size = sum_token_size
            else:
                break

        positive_prompt_str = ', '.join([prompt.Phrase for prompt in positive_prompt])
        if positive_prefix != "":
            positive_prompt_str = "{}, {}".format(positive_prefix, positive_prompt_str)
        negative_prompt_str = ', '.join([prompt.Phrase for prompt in negative_prompt])

        num_topics = len([prompt.Phrase for prompt in positive_prompt if "topic" in prompt.Types])
        num_modifiers = len([prompt.Phrase for prompt in positive_prompt if "modifier" in prompt.Types])
        num_styles = len([prompt.Phrase for prompt in positive_prompt if "style" in prompt.Types])
        num_constraints = len([prompt.Phrase for prompt in positive_prompt if "constraint" in prompt.Types])

        prompt = GeneratedPrompt(positive_prompt_str, negative_prompt_str, num_topics, num_modifiers,
                            num_styles, num_constraints, prompt_vector)
        # save prompt json
        prompt_json = prompt.to_json()
        filename = "{num:0{digits}}.json".format(num=count, digits=count_number_of_digits(prompt_count))
        file_path = os.path.join(dataset_output, filename)

        # Save the data to a JSON file
        with open(file_path, 'w') as json_file:
            json.dump(prompt_json, json_file)

        count += 1


def get_sorted_list_with_cumulative(phrases, phrases_token_size, count_list):
    # sort by count
    sorted_phrases = []
    sorted_token_size = []
    sorted_count = []
    sorted_cumulative_sum = []
    sorted_indexes = sorted(range(len(count_list)), key=lambda x: count_list[x], reverse=True)

    prev_sum = 0
    for i in sorted_indexes:
        sorted_phrases.append(phrases[i])
        sorted_token_size.append(phrases_token_size[i])
        sorted_count.append(count_list[i])

        # add cumulative sum
        cumulative_sum = prev_sum + count_list[i]
        sorted_cumulative_sum.append(cumulative_sum)
        prev_sum = cumulative_sum

    return sorted_phrases, sorted_token_size, sorted_count, sorted_cumulative_sum


def generate_prompts_from_csv_proportional_selection(csv_dataset_path,
                                                     csv_phrase_limit,
                                                     prompt_count,
                                                     positive_prefix="",
                                                     dataset_output=""):
    max_token_size = 75
    comma_token_size = 1

    phrases, \
        phrases_token_size,\
        positive_count_list,\
        negative_count_list = initialize_prompt_list_from_csv(csv_dataset_path, csv_phrase_limit)

    total_len_phrases = len(phrases)

    positive_phrases, \
        positive_token_size, \
        positive_count, \
        positive_cumulative_sum = get_sorted_list_with_cumulative(phrases, phrases_token_size, positive_count_list)

    positive_total_cumulative = positive_cumulative_sum[-1]

    negative_phrases, \
        negative_token_size, \
        negative_count, \
        negative_cumulative_sum = get_sorted_list_with_cumulative(phrases, phrases_token_size, negative_count_list)

    negative_total_cumulative = negative_cumulative_sum[-1]

    # del unused var at this point
    del phrases
    del phrases_token_size
    del positive_count_list
    del negative_count_list

    positive_prefix_token_size = 0
    if positive_prefix != "":
        # get token size for prefix
        enc = tiktoken.get_encoding("cl100k_base")
        positive_prefix_prompt_tokens = enc.encode(positive_prefix)
        positive_prefix_token_size = len(positive_prefix_prompt_tokens)

    print("Generating {} prompts...".format(prompt_count))
    count = 0
    for i in tqdm(range(0, prompt_count)):
        positive_prompt_total_token_size = positive_prefix_token_size
        negative_prompt_total_token_size = 0
        positive_prompt = []
        negative_prompt = []
        prompt_vector = [0] * total_len_phrases

        # positive prompt
        while positive_prompt_total_token_size < max_token_size:
            random_int = random.randint(0, positive_total_cumulative)
            random_index = find_first_element_binary_search(positive_cumulative_sum, random_int)
            if prompt_vector[random_index] != 0:
                continue

            prompt_index = random_index
            random_prompt = positive_phrases[prompt_index]

            chosen_phrase_size = positive_token_size[prompt_index]
            sum_token_size = positive_prompt_total_token_size + chosen_phrase_size + comma_token_size
            if sum_token_size < max_token_size:
                # update used array
                prompt_vector[prompt_index] = 1
                positive_prompt.append(random_prompt)
                positive_prompt_total_token_size = sum_token_size
            else:
                break

        # negative prompt
        while negative_prompt_total_token_size < max_token_size:
            random_int = random.randint(0, negative_total_cumulative)
            random_index = find_first_element_binary_search(negative_cumulative_sum, random_int)

            if prompt_vector[random_index] != 0:
                continue

            prompt_index = random_index
            random_prompt = negative_phrases[prompt_index]

            chosen_phrase_size = negative_token_size[prompt_index]
            sum_token_size = negative_prompt_total_token_size + chosen_phrase_size + comma_token_size
            if sum_token_size < max_token_size:
                # update used array
                prompt_vector[prompt_index] = -1
                negative_prompt.append(random_prompt)
                negative_prompt_total_token_size = sum_token_size
            else:
                break

        positive_prompt_str = ', '.join([prompt.Phrase for prompt in positive_prompt])
        if positive_prefix != "":
            positive_prompt_str = "{}, {}".format(positive_prefix, positive_prompt_str)
        negative_prompt_str = ', '.join([prompt.Phrase for prompt in negative_prompt])

        num_topics = len([prompt.Phrase for prompt in positive_prompt if "topic" in prompt.Types])
        num_modifiers = len([prompt.Phrase for prompt in positive_prompt if "modifier" in prompt.Types])
        num_styles = len([prompt.Phrase for prompt in positive_prompt if "style" in prompt.Types])
        num_constraints = len([prompt.Phrase for prompt in positive_prompt if "constraint" in prompt.Types])

        prompt = GeneratedPrompt(positive_prompt_str, negative_prompt_str, num_topics, num_modifiers,
                            num_styles, num_constraints, prompt_vector)

        # save prompt json
        prompt_json = prompt.to_json()
        filename = "{num:0{digits}}.json".format(num=count, digits=count_number_of_digits(prompt_count))
        file_path = os.path.join(dataset_output, filename)

        # Save the data to a JSON file
        with open(file_path, 'w') as json_file:
            json.dump(prompt_json, json_file)

        count += 1


# find the first element, whose cumulative total is more than the random number
def find_first_element_binary_search(cumulative_total_arr, random_num):
    low = 0
    high = len(cumulative_total_arr) - 1
    mid = 0

    while low <= high:
        mid = (high + low) / 2
        mid = math.floor(mid)

        # If random_num is greater, ignore left half
        if cumulative_total_arr[mid] < random_num:
            low = mid + 1
        # If random_num is smaller, ignore right half
        elif cumulative_total_arr[mid] > random_num:
            high = mid - 1
        # means random_num is present at mid
        else:
            return mid

        # use this index since sometimes the exact
        # random num is not in the list
        if low == high:
            return low

    # If we reach here, then the element was not present
    return -1


def count_number_of_digits(num):
    count = 0
    while (num > 0):
        count = count + 1
        num = num // 10
    return count


def generate_prompts_and_save_to_json(csv_dataset_path,
                                     csv_phrase_limit,
                                     prompt_count,
                                     positive_prefix="",
                                     dataset_output="",
                                     positive_ratio_threshold=3,
                                     negative_ratio_threshold=3,
                                     use_threshold=True,
                                     proportional_selection=False):
    # Create the directory if it doesn't exist
    if not os.path.exists(dataset_output):
        os.makedirs(dataset_output)

    if proportional_selection is True:
        generate_prompts_from_csv_proportional_selection(csv_dataset_path,
                                                        csv_phrase_limit,
                                                        prompt_count,
                                                        positive_prefix,
                                                        dataset_output)
    else:
        generate_prompts_from_csv(csv_dataset_path,
                                csv_phrase_limit,
                                prompt_count,
                                positive_prefix,
                                dataset_output,
                                positive_ratio_threshold,
                                negative_ratio_threshold,
                                use_threshold)

    shutil.make_archive(dataset_output, 'zip', dataset_output)
    print("Prompt list saved to {}".format(dataset_output))
    shutil.rmtree(dataset_output)