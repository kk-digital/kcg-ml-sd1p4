import os
import sys
import time

import torch

base_dir = os.getcwd()
sys.path.insert(0, base_dir)

import random
from os.path import join, abspath

import clip
import pygad
import argparse
import csv
import json

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

OUTPUT_DIR = abspath(join(base_dir, 'output', 'ga_1024'))

os.makedirs(EMBEDDED_PROMPTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Creating a new directory for this run of the GA (e.g. output/ga/ga001)
GA_RUN_DIR = get_next_ga_dir(OUTPUT_DIR)
os.makedirs(GA_RUN_DIR, exist_ok=True)

# Here we define IMAGES_ROOT_DIR and FEATURES_DIR based on the new GA_RUN_DIR
IMAGES_ROOT_DIR = os.path.join(GA_RUN_DIR, "images")
FEATURES_DIR = os.path.join(GA_RUN_DIR, "features")

os.makedirs(IMAGES_ROOT_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

csv_filename = os.path.join(GA_RUN_DIR, "fitness_data.csv")

# Write the headers to the CSV file
if not os.path.exists(csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Generation #', 'Population Size', 'Fitness (mean)', 'Fitness (variance)', 'Fitness (best)', 'Fitness array'])


fitness_cache = {}
start_time = time.time()

# TODO: NULL_PROMPT is completely wrong
NULL_PROMPT = None  # assign later

# DEVICE = input("Set device: 'cuda:i' or 'cpu'")
config = ModelPathConfig()

print(EMBEDDED_PROMPTS_DIR)
print(OUTPUT_DIR)
print(IMAGES_ROOT_DIR)
print(FEATURES_DIR)

# Initialize logger
def log_to_file(message):
    
    log_path = os.path.join(IMAGES_ROOT_DIR, "log.txt")

    with open(log_path, "a") as log_file:
        log_file.write(message + "\n")



# Function to calculate the chad score for batch of images
def calculate_and_store_images(ga_instance, solution, solution_idx):
    generation = ga_instance.generations_completed	
    # Set seed
    SEED = random.randint(0, 2**24)
    if FIXED_SEED == True:
        SEED = 54846

    # Calculate combined embedding
    combined_embedding_np = np.zeros((1, 77, 768))
    for i, coeff in enumerate(solution):
        combined_embedding_np += embedded_prompts_numpy[i] * coeff

    print(f"Generation {generation}, Solution {solution_idx}:")
    print(f"    Max value: {np.max(combined_embedding_np)}")
    print(f"    Min value: {np.max(combined_embedding_np)}")
    print(f"    Mean value: {np.mean(combined_embedding_np)}")
    print(f"    Standard deviation: {np.std(combined_embedding_np)}")    


    # Convert to PyTorch tensor and generate latent and image
    prompt_embedding = torch.tensor(combined_embedding_np, dtype=torch.float32).view(1, 77, 768).to(DEVICE)
    np.savez_compressed(os.path.join(FEATURES_DIR, f'prompt_embedding_{solution_idx}.npz'), prompt_embedding=prompt_embedding.cpu().numpy())
    latent = sd.generate_images_latent_from_embeddings(embedded_prompt=prompt_embedding, null_prompt=NULL_PROMPT, uncond_scale=CFG_STRENGTH)
    image = sd.get_image_from_latent(latent)

    # Save prompt and latent numpy arrays
    #np.savez_compressed(os.path.join(FEATURES_DIR, f'prompt_embedding_{solution_idx}.npz'), prompt_embedding=prompt_embedding.cpu().numpy())
    #np.savez_compressed(os.path.join(FEATURES_DIR, f'latent_{solution_idx}.npz'), latent=latent.cpu().numpy())

    # Convert to PIL image and calculate fitness score
    pil_image = to_pil(image[0])
    if CONVERT_GREY_SCALE_FOR_SCORING == True:
        pil_image = pil_image.convert("L").convert("RGB")
    
    fitness_score = white_background_fitness(pil_image)
    
    # Save the individual image
    file_dir = os.path.join(IMAGES_ROOT_DIR, str(generation))
    os.makedirs(file_dir, exist_ok=True)
    filename = os.path.join(file_dir, f'g{generation:04}_{solution_idx:04}.png')
    pil_image.save(filename)

    # Clean up to free memory
    del combined_embedding_np, prompt_embedding, latent, image, pil_image
    torch.cuda.empty_cache()

    return fitness_score



def cached_fitness_func(ga_instance, solution, solution_idx):
    solution_copy = solution.copy()  # flatten() is a destructive operation
    solution_flattened = solution_copy.flatten()
    solution_tuple = tuple(solution_flattened)

    if FIXED_SEED == True:
        # When FIXED_SEED is True, we try to get the score from the cache
        if solution_tuple in fitness_cache:
            print('Returning cached score', fitness_cache[solution_tuple])
            return fitness_cache[solution_tuple]
        else:
            # If it is not in the cache, we calculate it and then store it in the cache
            fitness_cache[solution_tuple] = calculate_and_store_images(ga_instance, solution, solution_idx)
            return fitness_cache[solution_tuple]
    else:
        # When FIXED_SEED is False, we do not use the cache and calculate a fresh score every time
        return calculate_and_store_images(ga_instance, solution, solution_idx)



def on_fitness(ga_instance, population_fitness):
    current_generation = ga_instance.generations_completed
    prompt_str = prompts_str_array[current_generation]
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


def on_start(ga_instance):
    log_to_file(f"Starting the genetic algorithm with {ga_instance.num_generations} generations and {ga_instance.sol_per_pop} population size.")

def on_generation(ga_instance):
    global start_time  # Make sure to define start_time as a global variable
    end_time = time.time()  # End timing for generation
    total_time = end_time - start_time
    log_to_file(f"----------------------------------")
    log_to_file(f"Total time taken for Generation #{ga_instance.generations_completed}: {total_time} seconds")

    # Log images per generation
    num_images = len(ga_instance.population)
    log_to_file(f"Images generated in Generation #{ga_instance.generations_completed}: {num_images}")

    # Log images/sec
    images_per_second = num_images / total_time
    log_to_file(f"Images per second in Generation #{ga_instance.generations_completed}: {images_per_second}")

    start_time = time.time()  # Reset the start time for the next generation


def clip_text_get_prompt_embedding_numpy(config, prompts: list):

    #load model from memory
    clip_text_embedder = CLIPTextEmbedder(device=get_device())
    clip_text_embedder.load_submodels(
        tokenizer_path=config.get_model_folder_path(CLIPconfigs.TXT_EMB_TOKENIZER),
        transformer_path=config.get_model_folder_path(CLIPconfigs.TXT_EMB_TEXT_MODEL)
    )

    prompt_embedding_numpy_list = []
    for prompt in prompts:
        print(prompt)
        prompt_embedding = clip_text_embedder.forward(prompt)
        prompt_embedding_cpu = prompt_embedding.cpu()

        del prompt_embedding
        torch.cuda.empty_cache()
        
        prompt_embedding_numpy_list.append(prompt_embedding_cpu.detach().numpy())
        # Flattening tensor and appending
        #print("clip_text_get_prompt_embedding, 1 embedding= ", str(torch.Tensor.size(prompt_embedding)))
        #clip_text_get_prompt_embedding, 1 embedding=  torch.Size([1, 77, 768])
        #prompt_embedding = prompt_embedding.view(-1)
        #print("clip_text_get_prompt_embedding, 2 embedding= ", str(torch.Tensor.size(prompt_embedding)))
        #clip_text_get_prompt_embedding, 2 embedding=  torch.Size([59136])


    ## Clear model from memory
    clip_text_embedder.to("cpu")
    del clip_text_embedder
    torch.cuda.empty_cache()

    return prompt_embedding_numpy_list


def prompt_embedding_vectors(sd, prompt_array):
    
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
num_genes = 1024 # Each individual is 1024 floats
prompt_phrase_length = 6  # number of words in prompt
prompts_array = ga.generate_prompts(num_genes, prompt_phrase_length)

# get prompt_str array
prompts_str_array = []
#prefix_prompt = " centered , white background, black object, black and white, no background,"
for prompt in prompts_array:
    prompt_str = prompt.get_positive_prompt_str()
    prompts_str_array.append(prompt_str)

# Construct the full path to the JSON file
json_file_path = os.path.join(IMAGES_ROOT_DIR, 'prompts_str_array.json')

# Saving the list to a JSON file
with open(json_file_path, 'w', encoding='utf-8') as f:
    json.dump(prompts_str_array, f, ensure_ascii=False, indent=4)

embedded_prompts_numpy = np.array(clip_text_get_prompt_embedding_numpy(config, prompts_str_array))


# random_mutation_min_val=5,
# random_mutation_max_val=10,
# mutation_by_replacement=True,

# note: uniform is good, two_points"

initial_population = []

for i in range(population_size):
    random_weights = np.random.normal(loc=0.0, scale=1.0, size=num_genes)
    random_weights = np.abs(random_weights)  # Making sure weights are non-negative
    random_weights /= random_weights.sum()  # Normalizing the weights to sum to 1
    initial_population.append(random_weights)

# Printing out each individual in the initial population
for i, individual in enumerate(initial_population):
    print(f"Individual {i}:")
    print(individual)

# Initialize the GA
ga_instance = pygad.GA(initial_population=initial_population,
                       num_generations=generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=cached_fitness_func,
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
                       on_generation=on_generation,
                       on_stop=on_fitness,
                       parent_selection_type=parent_selection_type,
                       keep_parents=0,
                       mutation_by_replacement=False,
                       random_mutation_min_val=-1,
                       random_mutation_max_val=1,
                       # fitness_func=calculate_fitness_score,
                       # on_parents=on_parents,
                       # on_crossover=on_crossover,
                       on_start=on_start,
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