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

import os
import sys
import time

base_dir = os.getcwd()
sys.path.insert(0, base_dir)

import random
from os.path import join

import clip
import pygad
import argparse
import csv 

# import safetensors as st

from chad_score.chad_score import ChadScorePredictor
from configs.model_config import ModelPathConfig
from stable_diffusion import StableDiffusion, SDconfigs
# TODO: rename stable_diffusion.utils_backend to /utils/cuda.py
from stable_diffusion.utils_backend import get_device
from stable_diffusion.utils_image import *
from ga.utils import get_next_ga_dir
import ga
from ga.fitness_chad_score import compute_chad_score_from_pil


random.seed()

N_STEPS = 20  # 20, 12
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
parser.add_argument('--mutation_percent_genes', type=float, default="0.001",
                    help="The percentage of genes to be mutated.")
args = parser.parse_args()

DEVICE = get_device()

# load clip
# get clip preprocessor
image_features_clip_model, preprocess = clip.load("ViT-L/14", device=DEVICE)

# load chad score
chad_score_model_path = os.path.join('input', 'model', 'chad_score', 'chad-score-v1.pth')
chad_score_predictor = ChadScorePredictor(device=DEVICE)
chad_score_predictor.load_model(chad_score_model_path)

# Why are you using this prompt generator?
EMBEDDED_PROMPTS_DIR = os.path.abspath(join(base_dir, 'input', 'embedded_prompts'))

OUTPUT_DIR = os.path.abspath(join(base_dir, 'output', 'ga'))
IMAGES_ROOT_DIR = os.path.abspath(join(OUTPUT_DIR, "images/"))
FEATURES_DIR = os.path.abspath(join(OUTPUT_DIR, "features/"))

os.makedirs(EMBEDDED_PROMPTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(IMAGES_ROOT_DIR, exist_ok=True)

# Creating new subdirectory for this run of the GA (e.g. output/ga/images/ga001)
IMAGES_DIR = get_next_ga_dir(IMAGES_ROOT_DIR)
os.makedirs(IMAGES_DIR, exist_ok=True)

csv_filename = os.path.join(IMAGES_DIR, "fitness_data.csv")

# Write the headers to the CSV file
if not os.path.exists(csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Generation #', 'Population Size', 'Fitness (mean)', 'Fitness (variance)', 'Fitness (best)', 'Fitness array'])

fitness_cache = {}

# TODO: NULL_PROMPT is completely wrong
NULL_PROMPT = None  # assign later

# DEVICE = input("Set device: 'cuda:i' or 'cpu'")
config = ModelPathConfig()

print(EMBEDDED_PROMPTS_DIR)
print(OUTPUT_DIR)
print(IMAGES_ROOT_DIR)
print(IMAGES_DIR)
print(FEATURES_DIR)

# TODO: wtf is this function
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

# Initialize logger
def log_to_file(message):
    
    log_path = os.path.join(IMAGES_DIR, "log.txt")

    with open(log_path, "a") as log_file:
        log_file.write(message + "\n")


# Function to calculate the chad score for batch of images
def calculate_chad_score(ga_instance, solution, solution_idx):
    # set seed
    SEED = random.randint(0, 2 ** 24)
    if FIXED_SEED == True:
        SEED = 54846

    # Convert the numpy array to a PyTorch tensor
    prompt_embedding = torch.tensor(solution, dtype=torch.float32)
    prompt_embedding = prompt_embedding.view(1, 77, 768).to(DEVICE)
    # print("embedded_prompt, tensor size, after= ",str(torch.Tensor.size(embedded_prompt)) )

    # print("Calculation Chad Score: sd.generate_images_from_embeddings")
    # print("prompt_embedded_prompt= " + str(prompt_embedding.get_device()))
    # print("null_prompt device= " + str(NULL_PROMPT.get_device()))
    # print("embedded_prompt, tensor size= ",str(torch.Tensor.size(prompt_embedding)) )
    # print("NULL_PROMPT, tensor size= ",str(torch.Tensor.size(NULL_PROMPT)) )
    # TODO: why are we regenerating the image?

    # NOTE: Is using NoGrad internally
    # NOTE: Is using autocast internally
    latent = sd.generate_images_latent_from_embeddings(
        seed=SEED,
        embedded_prompt=prompt_embedding,
        null_prompt=NULL_PROMPT,
        uncond_scale=CFG_STRENGTH
    )
    print("before deleting ! ")
    print(torch.cuda.memory_allocated() / 1024.0 / 1024.0)
    image = sd.get_image_from_latent(latent)
    del latent
    torch.cuda.empty_cache()
    print(torch.cuda.memory_allocated() / 1024.0 / 1024.0)
    print("after deleting ! ")

    # move back to cpu
    prompt_embedding.to("cpu")
    del prompt_embedding

    pil_image = to_pil(image[0])  # Convert to (height, width, channels)

    # convert to grey scale
    if CONVERT_GREY_SCALE_FOR_SCORING == True:
        pil_image = pil_image.convert("L")
        pil_image = pil_image.convert("RGB")

    _, chad_score = compute_chad_score_from_pil(pil_image)
    
    return chad_score


def cached_fitness_func(ga_instance, solution, solution_idx):
    solution_copy = solution.copy()  # flatten() is destructive operation
    solution_flattened = solution_copy.flatten()
    if tuple(solution_flattened) in fitness_cache:
        print('Returning cached score', fitness_cache[tuple(solution_flattened)])
    if tuple(solution_flattened) not in fitness_cache:
        fitness_cache[tuple(solution_flattened)] = calculate_chad_score(ga_instance, solution, solution_idx)
    return fitness_cache[tuple(solution_flattened)]


def on_fitness(ga_instance, population_fitness):
    population_fitness_np = np.array(population_fitness)
    print("Generation #", ga_instance.generations_completed)
    print("Population Size = ", len(population_fitness_np))
    print("Fitness (mean): ", np.mean(population_fitness_np))
    print("Fitness (variance): ", np.var(population_fitness_np))
    print("Fitness (best): ", np.max(population_fitness_np))
    print("fitness array= ", str(population_fitness_np))

    log_to_file(f"Generation #{ga_instance.generations_completed}")
    log_to_file(f"Population Size= {len(population_fitness_np)}")
    log_to_file(f"Fitness (mean): {np.mean(population_fitness_np)}")
    log_to_file(f"Fitness (variance): {np.var(population_fitness_np)}")
    log_to_file(f"Fitness (best): {np.max(population_fitness_np)}")
    log_to_file(f"fitness array= {str(population_fitness_np)}")

            # Append the current generation data to the CSV file
    with open(csv_filename, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([
            ga_instance.generations_completed,
            len(population_fitness_np),
            np.mean(population_fitness_np),
            np.var(population_fitness_np),
            np.max(population_fitness_np),
            str(population_fitness_np.tolist())
        ])
      
def on_mutation(ga_instance, offspring_mutation):
    print("Performing mutation at generation: ", ga_instance.generations_completed)
    log_to_file(f"Performing mutation at generation: {ga_instance.generations_completed}")


def store_generation_images(ga_instance):
    start_time = time.time()
    generation = ga_instance.generations_completed
    print("Generation #", generation)
    print("Population size: ", len(ga_instance.population))
    file_dir = os.path.join(IMAGES_DIR, str(generation))
    os.makedirs(file_dir)
    for i, ind in enumerate(ga_instance.population):
        SEED = random.randint(0, 2 ** 24)
        if FIXED_SEED == True:
            SEED = 54846
        prompt_embedding = torch.tensor(ind, dtype=torch.float32).to(DEVICE)
        prompt_embedding = prompt_embedding.view(1, 77, 768)

        print("prompt_embedding, tensor size= ", str(torch.Tensor.size(prompt_embedding)))
        print("NULL_PROMPT, tensor size= ", str(torch.Tensor.size(NULL_PROMPT)))

        # WARNING: Is using autocast internally
        latent = sd.generate_images_latent_from_embeddings(
            seed=SEED,
            embedded_prompt=prompt_embedding,
            null_prompt=NULL_PROMPT,
            uncond_scale=CFG_STRENGTH
        )

        image = sd.get_image_from_latent(latent)

        # move to gpu and cleanup
        prompt_embedding.to("cpu")
        del prompt_embedding

        pil_image = to_pil(image[0])
        filename = os.path.join(file_dir, f'g{generation:04}_{i:03}.png')
        pil_image.save(filename)

    end_time = time.time()  # End timing for generation
    total_time = end_time - start_time
    print(f"Total time taken for Generation #{generation}: {total_time} seconds")
    log_to_file(f"----------------------------------" )
    log_to_file(f"Total time taken for Generation #{generation}: {total_time} seconds")
    
    # Log images per generation
    num_images = len(ga_instance.population)
    print(f"Images generated in Generation #{generation}: {num_images}")
    log_to_file(f"Images generated in Generation #{generation}: {num_images}")
    
    # Log images/sec
    images_per_second = num_images / total_time
    print(f"Images per second in Generation #{generation}: {images_per_second}")
    log_to_file(f"Images per second in Generation #{generation}: {images_per_second}")


def prompt_embedding_vectors(sd, prompt_array):
    # Generate embeddings for each prompt
    embedded_prompts = ga.clip_text_get_prompt_embedding(config, prompts=prompt_array)
    # print("embedded_prompt, tensor shape= "+ str(torch.Tensor.size(embedded_prompts)))
    embedded_prompts.to("cpu")
    return embedded_prompts


# Call the GA loop function with your initialized StableDiffusion model

MUTATION_RATE = 0.01

generations = args.generations
population_size = 40
mutation_percent_genes = args.mutation_percent_genes
mutation_probability = args.mutation_probability
keep_elitism = args.keep_elitism

crossover_type = args.crossover_type
mutation_type = args.mutation_type
mutation_rate = 0.001

parent_selection_type = "tournament"  # "sss", rws, sus, rank, tournament

# num_parents_mating = int(population_size *.80)
num_parents_mating = int(population_size * .60)
keep_elitism = 0  # int(population_size*0.20)
mutation_probability = 0.10
# mutation_type = "adaptive" #try adaptive mutation
mutation_type = "swap"

'''
Random: mutation_type=random
Swap: mutation_type=swap
Inversion: mutation_type=inversion
Scramble: mutation_type=scramble
'''

# adaptive mutation:
# https://neptune.ai/blog/adaptive-mutation-in-genetic-algorithm-with-python-examples

# Load Stable Diffusion
sd = StableDiffusion(device=DEVICE, n_steps=N_STEPS)
sd.quick_initialize().load_autoencoder(config.get_model(SDconfigs.VAE)).load_decoder(config.get_model(SDconfigs.VAE_DECODER))
sd.model.load_unet(config.get_model(SDconfigs.UNET))
# calculate prompts

# Get embedding of null prompt
NULL_PROMPT = prompt_embedding_vectors(sd, [""])[0]
# print("NULL_PROMPT= ", str(NULL_PROMPT))
# print("NULL_PROMPT size= ", str(torch.Tensor.size(NULL_PROMPT)))

# generate prompts and get embeddings
prompt_phrase_length = 10  # number of words in prompt
prompts_array = ga.generate_prompts(population_size, prompt_phrase_length)

# get prompt_str array
prompts_str_array = []
for prompt in prompts_array:
    prompt_str = prompt.get_prompt_str()
    prompts_str_array.append(prompt_str)

embedded_prompts = prompt_embedding_vectors(sd, prompt_array=prompts_str_array)

print("genetic_algorithm_loop: population_size= ", population_size)

# ERROR: REVIEW
# TODO: What is this doing?
# Move the 'embedded_prompts' tensor to CPU memory
embedded_prompts_cpu = embedded_prompts.to("cpu")
embedded_prompts_array = embedded_prompts_cpu.detach().numpy()
embedded_prompts_list = embedded_prompts_array.reshape(population_size, 77 * 768).tolist()

# random_mutation_min_val=5,
# random_mutation_max_val=10,
# mutation_by_replacement=True,

# note: uniform is good, two_points"
crossover_type = "uniform"

num_genes = 77 * 768  # 59136
# Initialize the GA
ga_instance = pygad.GA(initial_population=embedded_prompts_list,
                       num_generations=generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=cached_fitness_func,
                       sol_per_pop=population_size,
                       num_genes=77 * 768,  # 59136
                       # Pygad uses 0-100 range for percentage
                       mutation_percent_genes=0.01,
                       # mutation_probability=mutation_probability,
                       mutation_probability=0.30,
                       keep_elitism=keep_elitism,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       on_fitness=on_fitness,
                       on_mutation=on_mutation,
                       on_generation=store_generation_images,
                       on_stop=on_fitness,
                       parent_selection_type=parent_selection_type,
                       keep_parents=0,
                       mutation_by_replacement=True,
                       random_mutation_min_val=5,
                       random_mutation_max_val=10,
                       # fitness_func=calculate_chad_score,
                       # on_parents=on_parents,
                       # on_crossover=on_crossover,
                       on_start=store_generation_images,
                       )
print(f"Batch Size: {population_size}")
print((f"Generations: {generations}"))
log_to_file(f"Batch Size: {population_size}")
log_to_file(f"Mutation Type: {mutation_type}")
log_to_file(f"Mutation Rate: {mutation_rate}")
log_to_file(f"Generations: {generations}")

ga_instance.run()

'''
Notes:
- 14 generatoins, readed 14 best
- with 12 rounds/iterations
- population size 16
- with uniform cross over
'''

del preprocess, image_features_clip_model, sd
