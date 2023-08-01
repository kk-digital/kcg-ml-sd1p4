
import random


class PromptGenerator:
    def __init__(self, word_list):
        self.word_list = word_list

    def random_prompt(self, n):
        this_prompt = ''
        this_prompt_list = []

        while n > 0:
            random_index = random.randint(0, len(self.word_list) - 1)
            chosen_string = self.word_list[random_index]
            if not chosen_string in this_prompt_list:
                this_prompt_list.append(chosen_string)
                n -= 1

        for prompt_item in this_prompt_list:
            if not this_prompt == '':
                this_prompt = this_prompt + ', '
            this_prompt = this_prompt + prompt_item

        return this_prompt