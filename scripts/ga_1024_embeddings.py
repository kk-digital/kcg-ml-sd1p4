import os
import sys
import time

import torch

base_dir = os.getcwd()
sys.path.insert(0, base_dir)

import random
from os.path import join

import clip
import pygad
import argparse
import csv

# import safetensors as st

from configs.model_config import ModelPathConfig
from stable_diffusion import StableDiffusion, SDconfigs
# TODO: rename stable_diffusion.utils_backend to /utils/cuda.py
from stable_diffusion.utils_backend import get_device
from stable_diffusion.utils_image import *
from ga.utils import get_next_ga_dir
import ga
from stable_diffusion import CLIPconfigs
from stable_diffusion.model.clip_text_embedder import CLIPTextEmbedder
#from ga.fitness_pixel_value import fitness_pixel_value
#from ga.fitness_white_background import white_background_fitness
from ga.fitness_filesize import filesize_fitness
from ga.fitness_white_background import white_background_fitness


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
parser.add_argument('--mutation_type', type=str, default="swap", help="Type of mutation operation.")
parser.add_argument('--mutation_percent_genes', type=float, default="0.001",
                    help="The percentage of genes to be mutated.")
args = parser.parse_args()

DEVICE = get_device()

# load clip
# get clip preprocessor
image_features_clip_model, preprocess = clip.load("ViT-L/14", device=DEVICE)


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

# Initialize logger
def log_to_file(message):
    
    log_path = os.path.join(IMAGES_DIR, "log.txt")

    with open(log_path, "a") as log_file:
        log_file.write(message + "\n")



# Function to calculate the chad score for batch of images
def calculate_fitness_score(ga_instance, solution, solution_idx):
    # Set seed
    SEED = random.randint(0, 2 ** 24)
    if FIXED_SEED == True:
        SEED = 54846

    combined_embedding_np = np.zeros((77, 768))
    for i, coeff in enumerate(solution):
        combined_embedding_np = combined_embedding_np + embedded_prompts_numpy[i] * coeff

    # Convert the combined numpy array to a PyTorch tensor
    prompt_embedding = torch.tensor(combined_embedding_np, dtype=torch.float32)
    prompt_embedding = prompt_embedding.view(1, 77, 768).to(DEVICE)
    
    # Generate image from the new combined_embedding
    latent = sd.generate_images_latent_from_embeddings(
        seed=SEED,
        embedded_prompt=prompt_embedding,
        null_prompt=NULL_PROMPT,
        uncond_scale=CFG_STRENGTH
    )
    
    # Create image from latent
    image = sd.get_image_from_latent(latent)

    # Move back to cpu and free the memory
    del combined_embedding_np

    prompt_embedding = prompt_embedding.to("cpu")
    del prompt_embedding


    pil_image = to_pil(image[0])  # Convert to (height, width, channels)

    # Convert to grey scale if needed
    if CONVERT_GREY_SCALE_FOR_SCORING == True:
        pil_image = pil_image.convert("L")
        pil_image = pil_image.convert("RGB")

    # Calculate fitness score
    fitness_score = white_background_fitness(pil_image)
    return fitness_score



def cached_fitness_func(ga_instance, solution, solution_idx):
    solution_copy = solution.copy()  # flatten() is destructive operation
    solution_flattened = solution_copy.flatten()
    if tuple(solution_flattened) in fitness_cache:
        print('Returning cached score', fitness_cache[tuple(solution_flattened)])
    if tuple(solution_flattened) not in fitness_cache:
        fitness_cache[tuple(solution_flattened)] = calculate_fitness_score(ga_instance, solution, solution_idx)
    return fitness_cache[tuple(solution_flattened)]


def on_fitness(ga_instance, population_fitness):
    population_fitness_np = np.array(population_fitness)
    print("Generation #", ga_instance.generations_completed)
    print("Population Size= ", len(population_fitness_np))
    print("Fitness (mean): ", np.mean(population_fitness_np))
    print("Fitness (variance): ", np.var(population_fitness_np))
    print("Fitness (best): ", np.max(population_fitness_np))
    print("Fitness array= ", str(population_fitness_np))

    log_to_file(f"Generation #{ga_instance.generations_completed}")
    log_to_file(f"Population Size= {len(population_fitness_np)}")
    log_to_file(f"Fitness (mean): {np.mean(population_fitness_np)}")
    log_to_file(f"Fitness (variance): {np.var(population_fitness_np)}")
    log_to_file(f"Fitness (best): {np.max(population_fitness_np)}")
    log_to_file(f"Prompt: {prompt_str}")
    log_to_file(f"Fitness array= {str(population_fitness_np)}")

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
        print(i)
        SEED = random.randint(0, 2 ** 24)
        if FIXED_SEED == True:
            SEED = 54846

        combined_embedding_np = np.zeros((77, 768))
        for ii, coeff in enumerate(ind):
            combined_embedding_np = combined_embedding_np + embedded_prompts_numpy[ii] * coeff

        combined_embedding = torch.from_numpy(combined_embedding_np)
        combined_embedding = combined_embedding.to(device=get_device(), dtype=torch.float32)
        # WARNING: Is using autocast internally
        latent = sd.generate_images_latent_from_embeddings(
            seed=SEED,
            embedded_prompt=combined_embedding,
            null_prompt=NULL_PROMPT,
            uncond_scale=CFG_STRENGTH
        )

        image = sd.get_image_from_latent(latent)
        del latent
        torch.cuda.empty_cache()

        pil_image = to_pil(image[0])
        del image
        torch.cuda.empty_cache()
        filename = os.path.join(file_dir, f'g{generation:04}_{i:04}.png')
        pil_image.save(filename)
        del pil_image  # Delete the PIL image
        torch.cuda.empty_cache()


    end_time = time.time()  # End timing for generation
    total_time = end_time - start_time
    log_to_file(f"----------------------------------" )
    log_to_file(f"Total time taken for Generation #{generation}: {total_time} seconds")
    
    # Log images per generation
    num_images = len(ga_instance.population)
    log_to_file(f"Images generated in Generation #{generation}: {num_images}")
    
    # Log images/sec
    images_per_second = num_images / total_time
    log_to_file(f"Images per second in Generation #{generation}: {images_per_second}")


def clip_text_get_prompt_embedding_numpy(config, prompts: list):
    #load model from memory
    clip_text_embedder = CLIPTextEmbedder(device=get_device())
    clip_text_embedder.load_submodels()

    prompt_embedding_numpy_list = []
    for prompt in prompts:
        print(prompt)
        prompt_embedding = clip_text_embedder(prompt)

        prompt_embedding_cpu = prompt_embedding.cpu()

        del prompt_embedding
        torch.cuda.empty_cache()

        prompt_embedding_numpy_list.append(prompt_embedding_cpu.detach().numpy())


    return prompt_embedding_numpy_list

def prompt_embedding_vectors(sd, prompt_array):
    # Generate embeddings for each prompt
    return clip_text_get_prompt_embedding_numpy(config, prompts=prompt_array)

# Call the GA loop function with your initialized StableDiffusion model


generations = args.generations
population_size = 12
mutation_percent_genes = args.mutation_percent_genes
mutation_probability = args.mutation_probability
keep_elitism = args.keep_elitism

crossover_type = args.crossover_type
mutation_type = args.mutation_type

parent_selection_type = "tournament"  # "sss", rws, sus, rank, tournament

# num_parents_mating = int(population_size *.80)
num_parents_mating = int(population_size * .60)
# mutation_type = "adaptive" #try adaptive mutation


# Load Stable Diffusion
sd = StableDiffusion(device=DEVICE, n_steps=N_STEPS)
sd.quick_initialize().load_autoencoder(config.get_model(SDconfigs.VAE)).load_decoder(config.get_model(SDconfigs.VAE_DECODER))
sd.model.load_unet(config.get_model(SDconfigs.UNET))
# calculate prompts

# Get embedding of null prompt
NULL_PROMPT = prompt_embedding_vectors(sd, [""])[0]
NULL_PROMPT = torch.from_numpy(NULL_PROMPT)
NULL_PROMPT = NULL_PROMPT.to(device=get_device(), dtype=torch.float32)
# print("NULL_PROMPT= ", str(NULL_PROMPT))
# print("NULL_PROMPT size= ", str(torch.Tensor.size(NULL_PROMPT)))

# generate prompts and get embeddings
num_genes = 1024  # Each individual is 1024 floats
prompt_phrase_length = 6  # number of words in prompt
prompts_array = ga.generate_prompts(num_genes, prompt_phrase_length)

# get prompt_str array
prompts_str_array = []
#prefix_prompt = " centered , white background, black object, black and white, no background,"
for prompt in prompts_array:
    prompt_str = prompt.get_positive_prompt_str()
    prompts_str_array.append(prompt_str)

print(prompt_str)

embedded_prompts_numpy = np.array(clip_text_get_prompt_embedding_numpy(config, prompts_str_array))


# random_mutation_min_val=5,
# random_mutation_max_val=10,
# mutation_by_replacement=True,

# note: uniform is good, two_points"

initial_population = []

for i in range(population_size):
    random_weights = np.random.dirichlet(np.ones(num_genes), size=1).flatten()
    #random_weights = np.full(num_prompts, 1.0 / num_prompts)
    #normalized_weights = (random_weights - np.mean(random_weights)) / np.std(random_weights)
    initial_population.append(random_weights)

# Initialize the GA
ga_instance = pygad.GA(initial_population=initial_population,
                       num_generations=generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=calculate_fitness_score,
                       sol_per_pop=population_size,
                       num_genes=num_genes, 
                       # Pygad uses 0-100 range for percentage
                       mutation_percent_genes= mutation_percent_genes,
                       mutation_probability = mutation_probability,
                       keep_elitism=keep_elitism,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       on_fitness=on_fitness,
                       on_mutation=on_mutation,
                       on_generation=store_generation_images,
                       on_stop=on_fitness,
                       parent_selection_type=parent_selection_type,
                       keep_parents=0,
                       mutation_by_replacement=False,
                       random_mutation_min_val=-1,
                       random_mutation_max_val=1,
                       # fitness_func=calculate_fitness_score,
                       # on_parents=on_parents,
                       # on_crossover=on_crossover,
                       on_start=store_generation_images,
                       )

log_to_file(f"Batch Size: {population_size}")
log_to_file(f"Mutation Type: {mutation_type}")
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
