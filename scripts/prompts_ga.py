
'''
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

#TODO: Using wrong function for getting device; use util
DEVICE = torch.device('cuda:0')

# Add argparse arguments
parser = argparse.ArgumentParser(description="Run genetic algorithm with specified parameters.")
parser.add_argument('--generations', type=int, default=2000, help="Number of generations to run.")
parser.add_argument('--mutation_probability', type=float, default=0.05, help="Probability of mutation.")
parser.add_argument('--keep_elitism', type=int, default=0, help="1 to keep best individual, 0 otherwise.")
parser.add_argument('--crossover_type', type=str, default="single_point", help="Type of crossover operation.")
parser.add_argument('--mutation_type', type=str, default="random", help="Type of mutation operation.")
parser.add_argument('--mutation_percent_genes', type=float, default="0.001", help="The percentage of genes to be mutated.")
args = parser.parse_args()

FIXED_SEED = True
CONVERT_GREY_SCALE_FOR_SCORING = True

#load clip
#get clip preprocessor
image_features_clip_model, preprocess = clip.load("ViT-L/14", device=DEVICE)

#load chad score
chad_score_model_path = 'input/model/chad_score/chad-score-v1.pth'
chad_score_predictor = ChadScorePredictor(device=DEVICE)
chad_score_predictor.load_model(chad_score_model_path)


#Why are you using this prompt generator?
EMBEDDED_PROMPTS_DIR = os.path.abspath(join(base_dir, "./input/embedded_prompts/"))

OUTPUT_DIR = os.path.abspath(os.path.join(base_dir, "./output/ga/"))
IMAGES_DIR = os.path.abspath(join(OUTPUT_DIR, "images/"))
FEATURES_DIR = os.path.abspath(join(OUTPUT_DIR, "features/"))

fitness_cache = {}

#TODO: NULL_PROMPT is completely wrong
NULL_PROMPT = torch.tensor(()).to(DEVICE)

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

# Function to calculate the chad score for batch of images
def calculate_chad_score(ga_instance, solution, solution_idx):


    # Copy the tensor to CUDA device if 'device' is 'cuda'
    # if device == 'cuda':
    #     solution_reshaped = solution_reshaped.to(device)

    # Generate an image using the solution

    SEED = random.randint(0, 2**24)
    if FIXED_SEED == True:
        SEED = 54846

    #image = generate_images_from_embeddings(solution_tensor, NULL_PROMPT)
    
    #WARNING: WTF is this line?
    #TODO: Delete next line, wtf

    # Convert the numpy array to a PyTorch tensor
    prompt_embedding = torch.tensor(solution, dtype=torch.float32)
    prompt_embedding = prompt_embedding.view(1, 77, 768).to(DEVICE)
    #print("embedded_prompt, tensor size, after= ",str(torch.Tensor.size(embedded_prompt)) )
 
    #TODO: why are we regenerating the image?
    #WARNING: Is not using NoGrad internally
    #WARNING: Is using autocast internally

    print("Calculation Chad Score: sd.generate_images_from_embeddings")
    #print("prompt_embedded_prompt= " + str(prompt_embedding.get_device()))
    #print("null_prompt device= " + str(NULL_PROMPT.get_device()))
    print("embedded_prompt, tensor size= ",str(torch.Tensor.size(prompt_embedding)) )
    print("NULL_PROMPT, tensor size= ",str(torch.Tensor.size(NULL_PROMPT)) ) 
    
    image = sd.generate_images_from_embeddings(
                seed=SEED,
                embedded_prompt=prompt_embedding,
                null_prompt=NULL_PROMPT
        )

    pil_image = to_pil(image[0])  # Convert to (height, width, channels)

    #convert to grey scale
    if CONVERT_GREY_SCALE_FOR_SCORING == True:
        pil_image = pil_image.convert("L")
        pil_image = pil_image.convert("RGB")

    unsqueezed_image = preprocess(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        image_features = image_features_clip_model.encode_image(unsqueezed_image)
        chad_score = chad_score_predictor.get_chad_score(image_features.type(torch.cuda.FloatTensor))
        #im_emb_arr = normalized(image_features.cpu().detach().numpy() )
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
    generation = ga_instance.generations_completed
    print("Generation #", generation)
    file_dir = os.path.join(IMAGES_DIR, str(generation))
    os.makedirs(file_dir)
    for i, ind in enumerate(ga_instance.population):
        SEED = random.randint(0, 2**24)
        if FIXED_SEED == True:
            SEED = 54846

        #WARNING: Is not using no grad internally
        #WARNING: Is using autocast internally
        #ERROR: dtype=torch.float16
        #image = generate_images_from_embeddings(torch.tensor(ind, dtype=torch.float16), NULL_PROMPT)
        prompt_embedding = torch.tensor(ind, dtype=torch.float32).to(DEVICE)
        prompt_embedding = embedded_prompt.view(1, 77, 768)

        #prompt_embedding.to(DEVICE)
        
        #embedded_prompt = torch.tensor(solution, dtype=torch.float32).to(DEVICE)

        #NULL_PROMPT.to(DEVICE)

        #torch.Tensor.get_device
        #print("prompt_embedding device= " + str(prompt_embedding.get_device()))
        #print("null_prompt device= " + str(NULL_PROMPT.get_device()))
        print("embedded_prompt, tensor size= ",str(torch.Tensor.size(prompt_embedding)) )
        print("NULL_PROMPT, tensor size= ",str(torch.Tensor.size(NULL_PROMPT)) ) 

        image = sd.generate_images_from_embeddings(
            seed=SEED,
            embedded_prompt=prompt_embedding,
            null_prompt=NULL_PROMPT
        )
        pil_image = to_pil(image[0])
        #TODO: g0000_000.png
        filename = os.path.join(file_dir, f'{i}.png')
        pil_image.save(filename)

def prompt_embedding_vectors(sd, prompt_count):
    PROMPT = ga.generate_prompts(prompt_count)
    #PROMPT = PROMPT[:prompt_count]


    # Generate embeddings for each prompt
    #embedded_prompts = get_prompt_clip_embedding(PROMPT)
    embedded_prompts = ga.clip_text_get_prompt_embedding(ModelConfig=pt, prompts=PROMPT)

    print("embedded_prompt, tensor shape= "+ str(torch.Tensor.size(embedded_prompts)))

    embedding = embedded_prompts
    #num_images = embedding.shape[0]
    #num_images = prompt_count

    '''
    embedded_prompts_cpu = embedded_prompts.cpu()
    embedded_prompts_array = embedded_prompts_cpu.detach().numpy()
    embedded_prompts_tensor = torch.tensor(embedded_prompts_array)
    '''

    #TODO: uhhh, wtf
    embedded_prompts_tensor = torch.tensor(embedded_prompts.cpu().detach().numpy())

    #num_individuals = embedded_prompts_array.shape[0]
    #num_individuals = prompt_count
    #num_genes = embedded_prompts_array.shape[1]
    #num_genes = 77*768
    
    return embedded_prompts_tensor


# Call the GA loop function with your initialized StableDiffusion model

N_STEPS = 20 #20, 12
CFG_STRENGTH = 9

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
embedded_prompts = prompt_embedding_vectors(sd, population_size)

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
                       on_generation=on_generation,
                       on_stop=on_fitness,
                       parent_selection_type= parent_selection_type,
                       keep_parents=0,
                       mutation_by_replacement=False,
                       random_mutation_min_val= -0.5, 
                       random_mutation_max_val= 0.5,
                       #fitness_func=calculate_chad_score,
                       #on_parents=on_parents,
                       #on_crossover=on_crossover,
                       #on_start=on_start,
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
