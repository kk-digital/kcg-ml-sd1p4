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

import torch

base_dir = os.getcwd()
sys.path.insert(0, base_dir)

import random
from os.path import join, abspath

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
from ga.fitness_chad_score import compute_chad_score_from_features


random.seed()

N_STEPS = 20  # 20, 12
CFG_STRENGTH = 9

FIXED_SEED = True
CONVERT_GREY_SCALE_FOR_SCORING = False

# Add argparse arguments
parser = argparse.ArgumentParser(description="Run genetic algorithm with specified parameters.")
parser.add_argument('--generations', type=int, default=2000, help="Number of generations to run.")
parser.add_argument('--population_size', type=int, default=100, help="Number of population to run.")
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

# TODO: NULL_PROMPT is completely wrong
NULL_PROMPT = None  # assign later

# DEVICE = input("Set device: 'cuda:i' or 'cpu'")
config = ModelPathConfig()

print(EMBEDDED_PROMPTS_DIR)
print(OUTPUT_DIR)
print(IMAGES_ROOT_DIR)
print(FEATURES_DIR)


clip_start_time = time.time()

# Initialize logger
def log_to_file(message):
    
    log_path = os.path.join(IMAGES_ROOT_DIR, "log.txt")

    with open(log_path, "a") as log_file:
        log_file.write(message + "\n")


def get_pil_image_from_solution(ga_instance, solution, solution_idx):
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

    return pil_image

def get_clip_features_from_pil(pil_image):
    '''
    Get CLIP features for a given PIL image.

    Args:
    - pil_image (PIL.Image.Image): Image for which CLIP features are to be computed.

    Returns:
    - feature_vector (torch.Tensor): CLIP feature vector for the input image.
    '''
    
    # Ensure the image is in RGB mode
    assert pil_image.mode == "RGB", "The image should be in RGB mode"

    # Preprocess the image and unsqueeze to add a batch dimension
    unsqueezed_image = preprocess(pil_image).unsqueeze(0).to(DEVICE)

    # Get CLIP encoding of the image
    with torch.no_grad():
        feature_vector = image_features_clip_model.encode_image(unsqueezed_image)
    
    return feature_vector

# Function to calculate the chad score for batch of images
def calculate_chad_score(ga_instance, solution, solution_idx):
    pil_image = get_pil_image_from_solution(ga_instance, solution, solution_idx)

    _, chad_score = compute_chad_score_from_features(pil_image)
    
    return chad_score


def cached_fitness_func(ga_instance, solution, solution_idx):
    if FIXED_SEED == True:
        # Use the cached score
        return ga_instance.population_fitness_list[solution_idx]
    else:
        # Get a fresh score because the seed is random
        return calculate_chad_score(ga_instance, solution, solution_idx)


def on_fitness(ga_instance, population_fitness):
    population_fitness_np = np.array(population_fitness)
    print("Generation #", ga_instance.generations_completed)
    print("Population Size = ", len(population_fitness_np))
    print("Fitness (mean): ", np.mean(population_fitness_np))
    print("Fitness (variance): ", np.var(population_fitness_np))
    print(f"Fitness (std): {np.sqrt(np.var(population_fitness_np))}") 
    print("Fitness (best): ", np.max(population_fitness_np))
    print("fitness array= ", str(population_fitness_np))

    log_to_file(f"Generation #{ga_instance.generations_completed}")
    log_to_file(f"Population Size= {len(population_fitness_np)}")
    log_to_file(f"Fitness (mean): {np.mean(population_fitness_np)}")
    log_to_file(f"Fitness (variance): {np.var(population_fitness_np)}")
    log_to_file(f"Fitness (std): {np.sqrt(np.var(population_fitness_np))}") 
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
    batch_size = ga_instance.batch_size
    population_fitness_list = []
    population_image_list = []
    print("Generation #", generation)
    print("Population size: ", len(ga_instance.population))
    file_dir = os.path.join(IMAGES_ROOT_DIR, str(generation))
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

        population_image_list.append(pil_image)

    num_images = len(population_image_list)
    num_batches = (num_images + batch_size - 1) // batch_size
    image_features = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_images)
        batch_images = population_image_list[start_idx:end_idx]

        batch_inputs = torch.stack([preprocess(image) for image in batch_images]).to(DEVICE)

        with torch.no_grad():
            batch_features = image_features_clip_model.encode_image(batch_inputs)

        for i, x in enumerate(batch_features):
            image_features.append(x)

    for i, image_feature in enumerate(image_features):
        with torch.no_grad():
            image_feature = image_feature.to(torch.float32)
            raw_chad_score = chad_score_predictor.get_chad_score(image_feature)

        scaled_chad_score = torch.sigmoid(torch.tensor(raw_chad_score))
        scaled_chad_score = scaled_chad_score.item()
        population_fitness_list.append(scaled_chad_score)

    ga_instance.population_fitness_list = population_fitness_list


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

generations = args.generations
mutation_percent_genes = args.mutation_percent_genes
mutation_probability = args.mutation_probability
keep_elitism = args.keep_elitism
population_size = args.population_size
crossover_type = args.crossover_type
mutation_type = args.mutation_type
mutation_rate = 0.001

parent_selection_type = "tournament"  # "sss", rws, sus, rank, tournament

# num_parents_mating = int(population_size *.80)
num_parents_mating = int(population_size * .60)
# mutation_type = "adaptive" #try adaptive mutation

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
    prompt_str = prompt.get_positive_prompt_str()
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

for idx, prompt_str in enumerate(prompts_str_array, 1):
    log_to_file(f"Prompt {idx}: {prompt_str}")

batch_size = 8
ga_instance.batch_size = batch_size

ga_instance.run()

# Record the end time after running the genetic algorithm
clip_end_time = time.time()

# Calculate the total time taken for Clip calculations
clip_total_time = clip_end_time - clip_start_time

# Get the number of Clip calculations made during the genetic algorithm
num_clip_calculations = len(ga_instance.population) * args.generations

# Calculate Clip calculations per second
clip_calculations_per_second = num_clip_calculations / clip_total_time

# Print the results
log_to_file(f"----------------------------------" )
log_to_file(f"Number of Clip Calculations: {num_clip_calculations} ")
log_to_file(f"Total Time for Clip Calculations: {clip_total_time} seconds")
log_to_file(f"Clip Calculations per Second {clip_calculations_per_second} ")

'''
Notes:
- 14 generatoins, readed 14 best
- with 12 rounds/iterations
- population size 16
- with uniform cross over
'''

del preprocess, image_features_clip_model, sd
