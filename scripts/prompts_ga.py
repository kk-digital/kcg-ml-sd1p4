import os, sys
base_dir = os.getcwd()
sys.path.insert(0, base_dir)

import torch
from typing import List
import configparser
import hashlib
import json
import math
import numpy as np
import random
from os.path import join

import clip
import pygad

#import safetensors as st

from chad_score.chad_score import ChadScorePredictor

from stable_diffusion import StableDiffusion
from stable_diffusion.model.clip_text_embedder import CLIPTextEmbedder
from stable_diffusion.model.clip_image_encoder import CLIPImageEncoder
from stable_diffusion.utils_model import *
from stable_diffusion.utils_backend import *
from stable_diffusion.utils_backend import get_device, get_memory_status
from stable_diffusion.utils_model import initialize_latent_diffusion
from stable_diffusion.utils_image import *
from stable_diffusion.constants import IODirectoryTree, create_directory_tree_folders
from stable_diffusion.constants import TOKENIZER_PATH, TEXT_MODEL_PATH
from transformers import CLIPTextModel, CLIPTokenizer

DEVICE = torch.device('cuda:0')
device = DEVICE
image_features_clip_model, preprocess = clip.load("ViT-L/14", device=device)
chad_score_predictor = ChadScorePredictor(device=device)
chad_score_predictor.load_model('input/model/chad_score/chad-score-v1.pth')

# Variables
#TODO: implement batch size
#BATCH_SIZE = 1

POPULATION_SIZE = 16
CFG_STRENGTH = 9
N_STEPS = 12 #20
GENERATIONS = 2000 #how many generations to run

MUTATION_RATE = 0.00

#Why are you using this prompt generator?
EMBEDDED_PROMPTS_DIR = os.path.abspath(join(base_dir, "./input/embedded_prompts/"))

OUTPUT_DIR = os.path.abspath(os.path.join(base_dir, "./output/ga/"))
IMAGES_DIR = os.path.abspath(join(OUTPUT_DIR, "images/"))
FEATURES_DIR = os.path.abspath(join(OUTPUT_DIR, "features/"))

fitness_cache = {}

# NULL_PROMPT is completely wrong
NULL_PROMPT = torch.tensor(()).to('cuda')

# DEVICE = input("Set device: 'cuda:i' or 'cpu'")
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(os.path.join(base_dir, "config.ini"))
config['BASE']['BASE_DIRECTORY'] = base_dir
config["BASE"].get('base_io_directory')

pt = IODirectoryTree(base_io_directory_prefix = config["BASE"].get('base_io_directory_prefix'), base_directory=base_dir)
pt.create_directory_tree_folders()

print(EMBEDDED_PROMPTS_DIR)
print(OUTPUT_DIR)
print(IMAGES_DIR)
print(FEATURES_DIR)

os.makedirs(EMBEDDED_PROMPTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

# Function to generate prompts
def generate_prompts(num_prompts):
    print("generate_prompts: Generating prompts")
    # List of prompt segments
    prompt_topics = [
    'chibi', 'waifu', 'cyborg', 'dragon', 'android', 'mecha', 
    'companion', 'furry', 'robot',
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
            prompt += random.choice(prompt_topics)
        for k in range(0, prompt_modifiers_count):
            prompt += random.choice(prompt_modifiers)
        prompt_list.append(prompt)

    print("prompt_list:")
    for i in range(0, len(prompt_list)):
        print("prompt ", i, ": ", prompt_list[i])
    
    return prompt_list

#def embed_and_save_prompts(prompts: list, null_prompt=NULL_PROMPT):
#def embed_and_save_prompts(prompts: list):
def get_prompt_clip_embedding(prompts: list):
    #null_prompt = null_prompt
    prompts = prompts

    clip_text_embedder = CLIPTextEmbedder(device=get_device())
    clip_text_embedder.load_submodels(**pt.embedder_submodels)

    embedded_prompts = []
    for prompt in prompts:
        embeddings = clip_text_embedder.forward(prompt)
        # Flattening tensor and appending
        embedded_prompts.append(embeddings.view(-1))
    embedded_prompts = torch.stack(embedded_prompts)
    torch.save(embedded_prompts, join(EMBEDDED_PROMPTS_DIR, "embedded_prompts.pt"))

    print(
        "Prompts embeddings saved at: ",
        f"{join(EMBEDDED_PROMPTS_DIR, 'embedded_prompts.pt')}",
    )

    get_memory_status()
    clip_text_embedder.to("cpu")
    del clip_text_embedder
    torch.cuda.empty_cache()
    get_memory_status()

    #return embedded_prompts, null_prompt
    return embedded_prompts

def normalized(a, axis=-1, order=2):
    import numpy as np

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def generate_images_from_embeddings(embedded_prompts_array, null_prompt):
    # print(embedded_prompts_array.to('cuda0'))
    SEED = random.randint(0, 2**24)
    #print("max_seed= ", 2**24)

    embedded_prompt = embedded_prompts_array.to('cuda').view(1, 77, 768)
    return sd.generate_images_from_embeddings(
        seed=SEED, embedded_prompt=embedded_prompt, null_prompt=null_prompt)

# Function to calculate the chad score for batch of images
def calculate_chad_score(ga_instance, solution, solution_idx):
    # Convert the numpy array to a PyTorch tensor
    solution_tensor = torch.tensor(solution, dtype=torch.float32)

    # Copy the tensor to CUDA device if 'device' is 'cuda'
    # if device == 'cuda':
    #     solution_reshaped = solution_reshaped.to(device)

    # Generate an image using the solution
    image = generate_images_from_embeddings(solution_tensor, NULL_PROMPT)

    pil_image = to_pil(image[0])  # Convert to (height, width, channels)
    unsqueezed_image = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = image_features_clip_model.encode_image(unsqueezed_image)
        chad_score = chad_score_predictor.get_chad_score(image_features.type(torch.cuda.FloatTensor))

        #im_emb_arr = normalized(image_features.cpu().detach().numpy() )
        #chad_score = chad_score_predictor.get_chad_score(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
        #chad_score = chad_score_predictor.get_chad_score(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))

        # chad_score = prediction.item()
        return chad_score

def cached_fitness_func(ga_instance, solution, solution_idx):
    if tuple(solution) in fitness_cache:
        print('Returning cached score', fitness_cache[tuple(solution)])
    if tuple(solution) not in fitness_cache:
        fitness_cache[tuple(solution)] = calculate_chad_score(ga_instance, solution, solution_idx)
    return fitness_cache[tuple(solution)]

def on_fitness(ga_instance, population_fitness):
    population_fitness_np = np.array(population_fitness)
    print("Generation #", ga_instance.generations_completed)
    print("Population Size= ", len(population_fitness_np))
    print("Fitness (mean): ", np.mean(population_fitness_np))
    print("Fitness (variance): ", np.var(population_fitness_np))
    print("Fitness (best): ", np.max(population_fitness_np))
    print("fitness array= ", str(population_fitness_np))

def on_mutation(ga_instance, offspring_mutation):
    print("Performing mutation at generation: ", ga_instance.generations_completed)

def on_generation(ga_instance):
    print("Completed one generation")
    generation = ga_instance.generations_completed
    file_dir = os.path.join(IMAGES_DIR, str(generation))
    os.makedirs(file_dir)
    for i, ind in enumerate(ga_instance.population):
        image = generate_images_from_embeddings(torch.tensor(ind, dtype=torch.float16), NULL_PROMPT)
        pil_image = to_pil(image[0])
        # filename=f"{IMAGES_DIR}/{}/{}.png"
        filename = os.path.join(file_dir, f'{i}.png')
        # pil_image.show()
        pil_image.save(filename)

# Define the GA loop function
def genetic_algorithm_loop(sd, embedded_prompts, null_prompt, generations, mutation_rate, population_size=POPULATION_SIZE,):
    
    print("genetic_algorithm_loop: population_size= ", population_size)

    # Move the 'embedded_prompts' tensor to CPU memory
    embedded_prompts_cpu = embedded_prompts.cpu()

    # Reshape the 'embedded_prompts' tensor to a 2D numpy array
    embedded_prompts_array = embedded_prompts_cpu.detach().numpy()
    num_individuals = embedded_prompts_array.shape[0]
    
    #num_genes = embedded_prompts_array.shape[1]
    num_genes = 77*768 #59136
    #print("num_genes= ", num_genes)

    embedded_prompts_list = embedded_prompts_array.reshape(num_individuals, num_genes).tolist()

    parent_selection_type = "tournament" #"sss", rws, sus, rank, tournament
    # Initialize the GA
    ga_instance = pygad.GA(num_generations=generations,
                           num_parents_mating=population_size/2,
                           # fitness_func=calculate_chad_score,
                           fitness_func=cached_fitness_func,
                           sol_per_pop=population_size,
                           num_genes=num_genes,
                           initial_population=embedded_prompts_list,
                           mutation_percent_genes=mutation_rate*100,
                           #on_start=on_start,
                           on_fitness=on_fitness,
                           #on_parents=on_parents,
                           #on_crossover=on_crossover,
                           on_mutation=on_mutation,
                           on_generation=on_generation,
                           on_stop=on_fitness,
                           parent_selection_type= parent_selection_type,
                           keep_parents=0
                           )

    '''
    keep_parents=-1 : Number of parents to keep in the current population.
    -1 (default) means keep all parents in the next population. 0 means keep no parents in the next population. A value greater than 0 means keep the specified number of parents in the next population. 
    '''
    ga_instance.keep_parents = 0
    ga_instance.num_parents_mating = 8

    ga_instance.run()
    return ga_instance.best_solution()


# Generate 6 random prompts with modifiers (initial population)
PROMPT = generate_prompts(POPULATION_SIZE)
#PROMPT = PROMPT[:20]
PROMPT = PROMPT[:POPULATION_SIZE]

# Load Stable Diffusion
sd = StableDiffusion(device=DEVICE, n_steps=N_STEPS)
sd.quick_initialize().load_autoencoder(**pt.autoencoder).load_decoder(**pt.decoder)
sd.model.load_unet(**pt.unet)

# Generate embeddings for each prompt
embedded_prompts = get_prompt_clip_embedding(PROMPT)

print("embedded_prompt, tensor shape= "+ str(torch.Tensor.size(embedded_prompts)))

embedding = embedded_prompts
num_images = embedding.shape[0]

embedded_prompts_cpu = embedded_prompts.cpu()
embedded_prompts_array = embedded_prompts_cpu.detach().numpy()
num_individuals = embedded_prompts_array.shape[0]
num_genes = embedded_prompts_array.shape[1]
embedded_prompts_tensor = torch.tensor(embedded_prompts_array)

# Call the GA loop function with your initialized StableDiffusion model
best_solution = genetic_algorithm_loop(sd, embedded_prompts_tensor, NULL_PROMPT, generations=GENERATIONS, mutation_rate=MUTATION_RATE)
print('best_solution', best_solution)

del preprocess, image_features_clip_model, sd
