
'''

TODO:
- generate N individuals
- then take latents as linear combination of best 64 (float 64 space)

TODO:
- GA search over noise seed for fixed prompt

TODO:
- compute clip, get gradient of chadscore, then go back to latent
- possible get gradient going back to embedding (impossible)
-- sampling process is non differentiable

TODO:
- GA for direct latent generation (64*64 = 4096)
- using chad score

TODO:
- GA for direct latent generation (64x64 = 4096)
- GA for matching input image (mean squared)

TODO:
- GA for direct latent generation (64x64 = 4096)
- GA for matching input image, clip score

TODO:
- GA for latent/bounding box size

TODO:
- two different clip functions
- custom scoring function
- jitters after crop

TODO:
- GA for searching for clip textt embedding (77*768)
- to minimized mean squared error against target image

TODO:
- GA for searching for clip text embedding (77*768)
- to maximize clip scores against a single image or set of images

TODO:
- operate on latent (64x64) with GA or with gradient

Objective:
- stable inversion
-- maximization should produce good images
- generate/rank
- 

'''

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
import argparse

#import safetensors as st

from chad_score.chad_score import ChadScorePredictor

from stable_diffusion import StableDiffusion
from stable_diffusion.model.clip_image_encoder import CLIPImageEncoder
from stable_diffusion.utils_model import *
#TODO: rename stable_diffusion.utils_backend to /utils/cuda.py
from stable_diffusion.utils_backend import *
from stable_diffusion.utils_backend import get_device, get_memory_status
from stable_diffusion.utils_model import initialize_latent_diffusion
from stable_diffusion.utils_image import *
from stable_diffusion.constants import IODirectoryTree, create_directory_tree_folders
from stable_diffusion.constants import TOKENIZER_PATH, TEXT_MODEL_PATH
#from transformers import CLIPTextModel, CLIPTokenizer

#from ga import generate_prompts
import ga

random.seed()

N_STEPS = 20 #20, 12
CFG_STRENGTH = 9

FIXED_SEED = True
CONVERT_GREY_SCALE_FOR_SCORING = False

# Add argparse arguments
parser = argparse.ArgumentParser(description="Run genetic algorithm with specified parameters.")
parser.add_argument('--generations', type=int, default=2000, help="Number of generations to run.")
parser.add_argument('--mutation_probability', type=float, default=0.05, help="Probability of mutation.")
parser.add_argument('--keep_elitism', type=int, default=0, help="1 to keep best individual, 0 otherwise.")
parser.add_argument('--crossover_type', type=str, default="single_point", help="Type of crossover operation.")
parser.add_argument('--mutation_type', type=str, default="random", help="Type of mutation operation.")
parser.add_argument('--mutation_percent_genes', type=float, default="0.001", help="The percentage of genes to be mutated.")
args = parser.parse_args()

DEVICE = get_device()

#load clip
#get clip preprocessor
image_features_clip_model, preprocess = clip.load("ViT-L/14", device=DEVICE)

#load chad score
chad_score_model_path = os.path.join('input', 'model', 'chad_score', 'chad-score-v1.pth')
chad_score_predictor = ChadScorePredictor(device=DEVICE)
chad_score_predictor.load_model(chad_score_model_path)


#Why are you using this prompt generator?
EMBEDDED_PROMPTS_DIR = os.path.abspath(join(base_dir, 'input', 'embedded_prompts'))

OUTPUT_DIR = os.path.abspath(join(base_dir, 'output', 'ga'))
IMAGES_DIR = os.path.abspath(join(OUTPUT_DIR, "images/"))
FEATURES_DIR = os.path.abspath(join(OUTPUT_DIR, "features/"))

fitness_cache = {}

#TODO: NULL_PROMPT is completely wrong
NULL_PROMPT = None #assign later

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

#TODO: wtf is this function
'''
def normalized(a, axis=-1, order=2):
    import numpy as np

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)
'''

'''
def generate_images_from_embeddings(embedded_prompts_array, null_prompt):
    # print(embedded_prompts_array.to('cuda0'))
    SEED = random.randint(0, 2**24)
    
    if FIXED_SEED == True:
        SEED = 54846
    #print("max_seed= ", 2**24)

    embedded_prompt = embedded_prompts_array.to('cuda').view(1, 77, 768)
    return sd.generate_images_from_embeddings(
        seed=SEED, embedded_prompt=embedded_prompt, null_prompt=null_prompt)
'''

def get_next_ga_dir(directory_path):
    num_digits = 3  # Number of digits for subdirectory names
    subdirectory_prefix = "ga"
    subdirectory_pattern = subdirectory_prefix + "{:0" + str(num_digits) + "}"

    existing_subdirs = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d)) and d.startswith(subdirectory_prefix)]

    if not existing_subdirs:
        # If no subdirectories exist, return "ga000"
        next_subdir_name = subdirectory_pattern.format(0)
    else:
        # Find the latest subdirectory name and extract the number part
        latest_subdir_name = max(existing_subdirs)
        latest_subdir_number = int(latest_subdir_name[2:])

        # Increment the number and format it with the specified number of digits
        next_subdir_number = latest_subdir_number + 1
        next_subdir_name = subdirectory_pattern.format(next_subdir_number)

    # Join the next subdirectory name with the input directory path
    next_subdirectory_path = os.path.join(directory_path, next_subdir_name)

    return next_subdirectory_path

# Function to calculate the chad score for batch of images
def calculate_chad_score(ga_instance, solution, solution_idx):
    #set seed
    SEED = random.randint(0, 2**24)
    if FIXED_SEED == True:
        SEED = 54846

    # Convert the numpy array to a PyTorch tensor
    prompt_embedding = torch.tensor(solution, dtype=torch.float32)
    prompt_embedding = prompt_embedding.view(1, 77, 768).to(DEVICE)
    #print("embedded_prompt, tensor size, after= ",str(torch.Tensor.size(embedded_prompt)) )

    #print("Calculation Chad Score: sd.generate_images_from_embeddings")
    #print("prompt_embedded_prompt= " + str(prompt_embedding.get_device()))
    #print("null_prompt device= " + str(NULL_PROMPT.get_device()))
    #print("embedded_prompt, tensor size= ",str(torch.Tensor.size(prompt_embedding)) )
    #print("NULL_PROMPT, tensor size= ",str(torch.Tensor.size(NULL_PROMPT)) ) 
    #TODO: why are we regenerating the image?

    #NOTE: Is using NoGrad internally
    #NOTE: Is using autocast internally
    image = sd.generate_images_from_embeddings(
        seed=SEED,
        embedded_prompt=prompt_embedding,
        null_prompt=NULL_PROMPT,
        uncond_scale=CFG_STRENGTH
    )
    #move back to cpu
    prompt_embedding.to("cpu")
    del prompt_embedding
    
    pil_image = to_pil(image[0])  # Convert to (height, width, channels)

    #convert to grey scale
    if CONVERT_GREY_SCALE_FOR_SCORING == True:
        pil_image = pil_image.convert("L")
        pil_image = pil_image.convert("RGB")

    unsqueezed_image = preprocess(pil_image).unsqueeze(0).to(DEVICE)
    #get clip encoding of model
    with torch.no_grad():
        image_features = image_features_clip_model.encode_image(unsqueezed_image)
        chad_score = chad_score_predictor.get_chad_score(image_features.type(torch.cuda.FloatTensor))
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

def store_generation_images(ga_instance):
    generation = ga_instance.generations_completed
    print("Generation #", generation)
    print("Population size: ", len(ga_instance.population))
    file_dir = os.path.join(IMAGES_DIR, str(generation))
    os.makedirs(file_dir)
    for i, ind in enumerate(ga_instance.population):
        SEED = random.randint(0, 2**24)
        if FIXED_SEED == True:
            SEED = 54846
        prompt_embedding = torch.tensor(ind, dtype=torch.float32).to(DEVICE)
        prompt_embedding = prompt_embedding.view(1, 77, 768)

        #print("prompt_embedding device= " + str(prompt_embedding.get_device()))
        #print("null_prompt device= " + str(NULL_PROMPT.get_device()))
        print("prompt_embedding, tensor size= ",str(torch.Tensor.size(prompt_embedding)) )
        print("NULL_PROMPT, tensor size= ",str(torch.Tensor.size(NULL_PROMPT)) ) 

        #WARNING: Is using autocast internally
        image = sd.generate_images_from_embeddings(
            seed=SEED,
            embedded_prompt=prompt_embedding,
            null_prompt=NULL_PROMPT,
            uncond_scale=CFG_STRENGTH
        )

        #move to gpu and cleanup
        prompt_embedding.to("cpu")
        del prompt_embedding

        pil_image = to_pil(image[0])
        #TODO: g0000_000.png
        filename = os.path.join(file_dir, f'{i}.png')
        pil_image.save(filename)

def prompt_embedding_vectors(sd, prompt_array):
    # Generate embeddings for each prompt
    embedded_prompts = ga.clip_text_get_prompt_embedding(ModelConfig=pt, prompts=prompt_array)
    #print("embedded_prompt, tensor shape= "+ str(torch.Tensor.size(embedded_prompts)))
    embedded_prompts.to("cpu")
    return embedded_prompts


# Creating new subdirectory for this run of the GA (e.g. output/ga/images/ga001)
os.makedirs(get_next_ga_dir(IMAGES_DIR))

# Call the GA loop function with your initialized StableDiffusion model

MUTATION_RATE = 0.01

generations = args.generations
population_size = 12
mutation_percent_genes = args.mutation_percent_genes
mutation_probability = args.mutation_probability
keep_elitism = args.keep_elitism

crossover_type = args.crossover_type
mutation_type = args.mutation_type
mutation_rate = 0.001

parent_selection_type = "tournament" #"sss", rws, sus, rank, tournament

#num_parents_mating = int(population_size *.80)
num_parents_mating = int(population_size *.25)
keep_elitism = 0 #int(population_size*0.20)
mutation_probability = 0.10
#mutation_type = "adaptive" #try adaptive mutation
mutation_type="swap"

'''
Random: mutation_type=random
Swap: mutation_type=swap
Inversion: mutation_type=inversion
Scramble: mutation_type=scramble
'''

#adaptive mutation: 
#https://neptune.ai/blog/adaptive-mutation-in-genetic-algorithm-with-python-examples

# Load Stable Diffusion
sd = StableDiffusion(device=DEVICE, n_steps=N_STEPS)
sd.quick_initialize().load_autoencoder(**pt.autoencoder).load_decoder(**pt.decoder)
sd.model.load_unet(**pt.unet)
#calculate prompts

#Get embedding of null prompt
NULL_PROMPT = prompt_embedding_vectors(sd,[""])[0]
#print("NULL_PROMPT= ", str(NULL_PROMPT))
#print("NULL_PROMPT size= ", str(torch.Tensor.size(NULL_PROMPT)))

#generate prompts and get embeddings
prompts_array = ga.generate_prompts(population_size)
embedded_prompts = prompt_embedding_vectors(sd, prompt_array=prompts_array)

print("genetic_algorithm_loop: population_size= ", population_size)

#ERROR: REVIEW
#TODO: What is this doing?
# Move the 'embedded_prompts' tensor to CPU memory
embedded_prompts_cpu = embedded_prompts.cpu()
embedded_prompts_array = embedded_prompts_cpu.detach().numpy()
embedded_prompts_list = embedded_prompts_array.reshape(population_size, 77*768).tolist()

#random_mutation_min_val=5,
#random_mutation_max_val=10,
#mutation_by_replacement=True,

#note: uniform is good, two_points"
crossover_type = "uniform"

num_genes = 77*768 #59136
# Initialize the GA
ga_instance = pygad.GA(initial_population=embedded_prompts_list,
                       num_generations=generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=cached_fitness_func,
                       sol_per_pop=population_size,
                       num_genes=77*768, #59136
                       # Pygad uses 0-100 range for percentage
                       mutation_percent_genes=0.01,
                       #mutation_probability=mutation_probability,
                       mutation_probability=0.30,
                       keep_elitism=keep_elitism,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       on_fitness=on_fitness,
                       on_mutation=on_mutation,
                       on_generation=store_generation_images,
                       on_stop=on_fitness,
                       parent_selection_type= parent_selection_type,
                       keep_parents=0,
                       mutation_by_replacement=True,
                       random_mutation_min_val= 5,
                       random_mutation_max_val=10,
                       #fitness_func=calculate_chad_score,
                       #on_parents=on_parents,
                       #on_crossover=on_crossover,
                       on_start=store_generation_images,
                       )

ga_instance.run()

'''
Notes:
- 14 generatoins, readed 14 best
- with 12 rounds/iterations
- population size 16
- with uniform cross over
'''

del preprocess, image_features_clip_model, sd
