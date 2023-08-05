#prompt_generator.py

import random

# Function to generate prompts
def generate_prompts(num_prompts):
    print("generate_prompts: Generating prompts")
    # List of prompt segments
    prompt_topics = [
    'chibi', 'waifu', 'cyborg', 'dragon', 'android', 'mecha', 
    'companion', 'furry', 'robot',
    'mercentary', 'wizard', 'pet', 
    'mercentary', 'wizard', 'pet',
    'shapeshifter', 'pilot', 'time traveler', "engineer", "slaver",
    ]

    # Add modifiers to the selected prompts
    prompt_modifiers = [
        'beautiful', 'unreal', 'masterpiece', 'gorgeous', 'stunning',
        'captivating', 'breathtaking',
        'exquisite', 'magnificent', 'majestic', 'elegant', 'sublime',
        'futuristic', 'cyberpunk', 'hi-tech', 'advanced', 'innovative', 'modern',
        'fantasy', 'mythical', 'scifi', 'side scrolling', 'character', 'side scrolling',
        'white background', 'centered', 'full character', 'no background', 'not centered',
        'line drawing', 'sketch', 'black and white', 'colored','video game'
        ]

    prompt_list = []
    prompt_base = "side scrolling, chibi, waifu, centered, white background, "
    prompt_topic_count = 2
    prompt_modifiers_count = 8

    for i in range(0, num_prompts):
        prompt = prompt_base    
        for j in range(0, prompt_topic_count):
            prompt = prompt + ", " + random.choice(prompt_topics)
        for k in range(0, prompt_modifiers_count):
            prompt = prompt + ", " + random.choice(prompt_modifiers)
        prompt_list.append(prompt)

    print("prompt_list:")
    for i in range(0, len(prompt_list)):
        print("prompt ", i, ": ", prompt_list[i])
    
    return prompt_list