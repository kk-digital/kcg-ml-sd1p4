# prompt_generator.py
import random
import tiktoken
import time

class GeneratedPrompt():
    def __init__(self, prompt_dict: [], prompt_vector: []):
        self.prompt_dict = prompt_dict
        self.prompt_str = ', '.join([prompt.Phrase for prompt in prompt_dict])
        self.num_topics = len([prompt.Phrase for prompt in prompt_dict if "topic" in prompt.Types])
        self.num_modifiers = len([prompt.Phrase for prompt in prompt_dict if "modifier" in prompt.Types])
        self.num_styles = len([prompt.Phrase for prompt in prompt_dict if "style" in prompt.Types])
        self.num_constraints = len([prompt.Phrase for prompt in prompt_dict if "constraint" in prompt.Types])

        # prompt_vector is a vector of 0 or 1, 1 if that index of prompt list is used
        self.prompt_vector = prompt_vector

class PromptData():
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

    def add_phrase(self, phrase: str):
        index = len(self.Prompts)
        new_prompt = PromptData(index, phrase)
        self.Prompts.append(new_prompt)

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
            is_prompt_type_exists = len([prompt_type for prompt_type in prompt_data.Types if prompt_type == prompt_type]) > 0
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


def initialize_prompt_list():
    prompt_list = PromptList()

    prompt_list.add_phrases(
        ["chibi", "waifu", "cyborg", "dragon", "android", "mecha", "companion", "furry", "robot", "mercenary", "wizard",
         "pet", "wizard", "shapeshifter", "pilot", "time traveler", "engineer", "slaver", "samurai"])

    prompt_list.add_phrases(
        ["unreal", "masterpiece", "gorgeous", "stunning", "captivating", "breathtaking", "exquisite", "magnificent",
         "majestic", "elegant", "sublime", "futuristic", "cyberpunk", "hi-tech", "advanced", "innovative", "modern",
         "fantasy", "mythical", "scifi", "side scrolling", "character", "white background", "centered",
         "full character", "no background", "not centered", "line drawing", "sketch", "black and white", "colored",
         "video game"])

    prompt_list.add_type_to_phrase("chibi", "topic")
    prompt_list.add_type_to_phrase("waifu", "topic")
    prompt_list.add_type_to_phrase("cyborg", "topic")
    prompt_list.add_type_to_phrase("dragon", "topic")
    prompt_list.add_type_to_phrase("android", "topic")
    prompt_list.add_type_to_phrase("mecha", "topic")
    prompt_list.add_type_to_phrase("companion", "topic")
    prompt_list.add_type_to_phrase("furry", "topic")
    prompt_list.add_type_to_phrase("robot", "topic")
    prompt_list.add_type_to_phrase("mercenary", "topic")
    prompt_list.add_type_to_phrase("wizard", "topic")
    prompt_list.add_type_to_phrase("pet", "topic")
    prompt_list.add_type_to_phrase("shapeshifter", "topic")
    prompt_list.add_type_to_phrase("pilot", "topic")
    prompt_list.add_type_to_phrase("time traveler", "topic")
    prompt_list.add_type_to_phrase("engineer", "topic")
    prompt_list.add_type_to_phrase("slaver", "topic")
    prompt_list.add_type_to_phrase("samurai", "topic")

    prompt_list.add_type_to_phrase("beautiful", "modifier")
    prompt_list.add_type_to_phrase("unreal", "modifier")
    prompt_list.add_type_to_phrase("masterpiece", "modifier")
    prompt_list.add_type_to_phrase("gorgeous", "modifier")
    prompt_list.add_type_to_phrase("stunning", "modifier")
    prompt_list.add_type_to_phrase("captivating", "modifier")
    prompt_list.add_type_to_phrase("breathtaking", "modifier")
    prompt_list.add_type_to_phrase("exquisite", "modifier")
    prompt_list.add_type_to_phrase("magnificent", "modifier")
    prompt_list.add_type_to_phrase("majestic", "modifier")
    prompt_list.add_type_to_phrase("elegant", "modifier")
    prompt_list.add_type_to_phrase("sublime", "modifier")
    prompt_list.add_type_to_phrase("futuristic", "modifier")
    prompt_list.add_type_to_phrase("cyberpunk", "modifier")
    prompt_list.add_type_to_phrase("hi-tech", "modifier")
    prompt_list.add_type_to_phrase("advanced", "modifier")
    prompt_list.add_type_to_phrase("innovative", "modifier")
    prompt_list.add_type_to_phrase("modern", "modifier")
    prompt_list.add_type_to_phrase("fantasy", "modifier")
    prompt_list.add_type_to_phrase("mythical", "modifier")
    prompt_list.add_type_to_phrase("scifi", "modifier")
    prompt_list.add_type_to_phrase("character", "modifier")
    prompt_list.add_type_to_phrase("full character", "modifier")
    prompt_list.add_type_to_phrase("black and white", "modifier")
    prompt_list.add_type_to_phrase("colored", "modifier")
    prompt_list.add_type_to_phrase("video game", "modifier")

    prompt_list.add_type_to_phrase("white background", "constraint")
    prompt_list.add_type_to_phrase("no background", "constraint")
    prompt_list.add_type_to_phrase("not centered", "constraint")
    prompt_list.add_type_to_phrase("centered", "constraint")

    prompt_list.add_type_to_phrase("sketch", "style")
    prompt_list.add_type_to_phrase("line drawing", "style")
    prompt_list.add_type_to_phrase("side scrolling", "style")

    return prompt_list.Prompts


prompts = initialize_prompt_list()


# Function to generate prompts
def generate_prompts(prompt_phrase_length, num_phrases):
    print("generate_prompts: Generating prompts")

    prompt_list = []
    enc = tiktoken.get_encoding("cl100k_base")

    for i in range(0, prompt_phrase_length):
        num_tokens = 100
        while num_tokens > 77:
            prompt_dict = []
            prompt_vector = [0] * len(prompts)
            for j in range(0, num_phrases):
                random_prompt = random.choice(
                    [item for item in prompts if (prompt_vector[item.Index] == 0)])
                prompt_index = random_prompt.Index

                # update used array
                prompt_vector[prompt_index] = 1
                prompt_dict.append(random_prompt)

            prompt_str = ', '.join([prompt.Phrase for prompt in prompt_dict])

            start = time.time()
            print("Calculating number of tokens for prompt: {0}".format(prompt_str))

            # check the length of prompt embedding.
            # if it's more than 77, then regenerate/reroll
            # (max token size is 77)
            prompt_tokens = enc.encode(prompt_str)
            num_tokens = len(prompt_tokens)

            print("Number of tokens: {0}".format(num_tokens))
            print("Elapsed Time: {0}s".format(time.time() - start))
            print("-------------------------------------------------------------------------")


        prompt_list.append(GeneratedPrompt(prompt_dict, prompt_vector))

    print("prompt_list:")
    for i in range(0, len(prompt_list)):
        print("prompt ", i, ": prompt_str:", prompt_list[i].prompt_str)
        print("prompt ", i, ": number of topics:", prompt_list[i].num_topics)
        print("prompt ", i, ": number of modifiers:", prompt_list[i].num_modifiers)
        print("prompt ", i, ": number of styles:", prompt_list[i].num_styles)
        print("prompt ", i, ": number of constraints:", prompt_list[i].num_constraints)
        print("prompt ", i, ": prompt_vector:", prompt_list[i].prompt_vector)
        print("-------------------------------------------------------------------------")

    return prompt_list
