import os
import sys
import time

base_dir = os.getcwd()
sys.path.insert(0, base_dir)

import random
from os.path import join, abspath

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
from ga.fitness_bounding_box_centered import centered_fitness



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



# Why are you using this prompt generator?
EMBEDDED_PROMPTS_DIR = os.path.abspath(join(base_dir, 'input', 'embedded_prompts'))

OUTPUT_DIR = abspath(join(base_dir, 'output', 'ga_bounding_box'))

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
def get_pil_image_from_solution(ga_instance, solution, solution_idx):
    

    # set seed
    SEED = random.randint(0, 2 ** 24)
    if FIXED_SEED == True:
        SEED = 54846

    # Convert the numpy array to a PyTorch tensor
    prompt_embedding = torch.tensor(solution, dtype=torch.float32)
    prompt_embedding = prompt_embedding.view(1, 77, 768).to(DEVICE)

    latent = sd.generate_images_latent_from_embeddings(
        seed=SEED,
        embedded_prompt=prompt_embedding,
        null_prompt=NULL_PROMPT,
        uncond_scale=CFG_STRENGTH
    )

    image = sd.get_image_from_latent(latent)
    del latent
    torch.cuda.empty_cache()

    # move back to cpu
    prompt_embedding.to("cpu")
    del prompt_embedding

    pil_image = to_pil(image[0])  # Convert to (height, width, channels)
    del image
    torch.cuda.empty_cache()

    # convert to grey scale
    if CONVERT_GREY_SCALE_FOR_SCORING == True:
        pil_image = pil_image.convert("L")
        pil_image = pil_image.convert("RGB")

    # Save the generated image
    generation = ga_instance.generations_completed
    file_dir = os.path.join(IMAGES_ROOT_DIR, str(generation))
    os.makedirs(file_dir, exist_ok=True)
    filename = os.path.join(file_dir, f'g{generation:04}_{solution_idx:03}.png')
    pil_image.save(filename)

    return pil_image


# Function to calculate the chad score for batch of images
def calculate_fitness_score(ga_instance, solution, solution_idx):
    pil_image = get_pil_image_from_solution(ga_instance, solution, solution_idx)

    fitness_score = centered_fitness(pil_image)
    
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
            fitness_cache[solution_tuple] = calculate_fitness_score(ga_instance, solution, solution_idx)
            return fitness_cache[solution_tuple]
    else:
        # When FIXED_SEED is False, we do not use the cache and calculate a fresh score every time
        return calculate_fitness_score(ga_instance, solution, solution_idx)


def on_fitness(ga_instance, population_fitness):
    population_fitness_np = np.array(population_fitness)
    print("Generation #", ga_instance.generations_completed)
    print("Population Size= ", len(population_fitness_np))
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




def prompt_embedding_vectors(sd, prompt_array):
    # Generate embeddings for each prompt
    embedded_prompts = ga.clip_text_get_prompt_embedding(config, prompts=prompt_array)
    embedded_prompts.to("cpu")
    return embedded_prompts


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

parent_selection_type = "tournament"  # "sss", rws, sus, rank, tournament

# num_parents_mating = int(population_size *.80)
num_parents_mating = int(population_size * .60)
keep_elitism = 0  # int(population_size*0.20)
mutation_probability = 0.10
# mutation_type = "adaptive" #try adaptive mutation
mutation_type = "swap"


# Load Stable Diffusion
sd = StableDiffusion(device=DEVICE, n_steps=N_STEPS)
sd.quick_initialize().load_autoencoder(config.get_model(SDconfigs.VAE)).load_decoder(config.get_model(SDconfigs.VAE_DECODER))
sd.model.load_unet(config.get_model(SDconfigs.UNET))
# calculate prompts

# Get embedding of null prompt
NULL_PROMPT = prompt_embedding_vectors(sd, [""])[0]

# generate prompts and get embeddings
prompt_phrase_length = 10  # number of words in prompt
prompts_array = ga.generate_prompts(population_size, prompt_phrase_length)

# get prompt_str array
prompts_str_array = []
prefix_prompt = " centered , forth of image, black character, white background, black object, black and white, no background,"
for prompt in prompts_array:
    prompt_str = prefix_prompt + prompt.get_positive_prompt_str()
    prompts_str_array.append(prompt_str)

print(prompt_str)
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
                       on_generation=on_generation,
                       on_stop=on_fitness,
                       parent_selection_type=parent_selection_type,
                       keep_parents=0,
                       mutation_by_replacement=True,
                       random_mutation_min_val=5,
                       random_mutation_max_val=10,
                       # fitness_func=calculate_fitness_score,
                       # on_parents=on_parents,
                       # on_crossover=on_crossover,
                       on_start=on_start,
                       )

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
del sd
