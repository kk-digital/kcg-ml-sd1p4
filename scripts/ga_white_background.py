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

from configs.model_config import ModelPathConfig
from stable_diffusion import StableDiffusion, SDconfigs
# TODO: rename stable_diffusion.utils_backend to /utils/cuda.py
from stable_diffusion.utils_backend import get_device
from stable_diffusion.utils_image import *
from ga.utils import get_next_ga_dir
import ga
from ga.fitness_pixel_value import fitness_pixel_value
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
parser.add_argument('--mutation_type', type=str, default="random", help="Type of mutation operation.")
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

    fitness_score = white_background_fitness(pil_image)
    #fitness_pixel = fitness_pixel_value(pil_image)

    #fitness_score = (fitness_white + fitness_pixel) / 2.0

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
        del latent
        torch.cuda.empty_cache()

        # move to gpu and cleanup
        prompt_embedding.to("cpu")
        del prompt_embedding

        pil_image = to_pil(image[0])
        del image
        torch.cuda.empty_cache()
        filename = os.path.join(file_dir, f'g{generation:04}_{i:03}.png')
        pil_image.save(filename)


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


def prompt_embedding_vectors(sd, prompt_array):
    # Generate embeddings for each prompt
    embedded_prompts = ga.clip_text_get_prompt_embedding(config, prompts=prompt_array)
    # print("embedded_prompt, tensor shape= "+ str(torch.Tensor.size(embedded_prompts)))
    embedded_prompts.to("cpu")
    return embedded_prompts


# Call the GA loop function with your initialized StableDiffusion model


generations = args.generations
population_size = 64
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
# print("NULL_PROMPT= ", str(NULL_PROMPT))
# print("NULL_PROMPT size= ", str(torch.Tensor.size(NULL_PROMPT)))

# generate prompts and get embeddings
prompt_phrase_length = 6  # number of words in prompt
prompts_array = ga.generate_prompts(population_size, prompt_phrase_length)

# get prompt_str array
prompts_str_array = []
prefix_prompt = "isolated on white background, simple background, background chroma plain white, small character, on white background, solid color background no white character, on a white surface, tiny character,  centered , small character"
# Append prefix_prompt to prompts_str_array
prompts_str_array.append(prefix_prompt)

prefix_prompt1 = "isolated on white background, simple background, background chroma plain white, small character "


# Use a range of 63 since you've already added one prompt
for i in range(63):
    prompt_str = prefix_prompt1 + ", " +  prompts_array[i].get_prompt_str()
    prompts_str_array.append(prompt_str)

print(prompt_str)
embedded_prompts = prompt_embedding_vectors(sd, prompt_array=prompts_str_array)

print("genetic_algorithm_loop: population_size= ", population_size)

# ERROR: REVIEW
# TODO: What is this doing?
# Move the 'embedded_prompts' tensor to CPU memory
embedded_prompts_cpu = embedded_prompts.to("cpu")
embedded_prompts_array = embedded_prompts_cpu.detach().numpy()
print(f"array shape: {embedded_prompts_array.shape}")
embedded_prompts_list = embedded_prompts_array.reshape(population_size, 77 * 768).tolist()

# random_mutation_min_val=5,
# random_mutation_max_val=10,
# mutation_by_replacement=True,

# note: uniform is good, two_points"

num_genes = 77 * 768  # 59136
# Initialize the GA
ga_instance = pygad.GA(initial_population=embedded_prompts_list,
                       num_generations=generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=cached_fitness_func,
                       sol_per_pop=population_size,
                       num_genes=77 * 768,  # 59136
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
                       mutation_by_replacement=True,
                       random_mutation_min_val=5,
                       random_mutation_max_val=10,
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
