import os
import sys
import time

base_dir = os.getcwd()
sys.path.insert(0, base_dir)

import random
from os.path import join
import shutil
import clip
import pygad
import argparse
import csv

from chad_score.chad_score import ChadScorePredictor
from model.util_clip import UtilClip
from configs.model_config import ModelPathConfig
from stable_diffusion import StableDiffusion, SDconfigs
# TODO: rename stable_diffusion.utils_backend to /utils/cuda.py
from stable_diffusion.utils_backend import get_device
from stable_diffusion.utils_image import *
from ga.utils import get_next_ga_dir
import ga
from ga.fitness_chad_score import compute_chad_score_from_pil


def parse_args():
    # Add argparse arguments
    parser = argparse.ArgumentParser(description="Run genetic algorithm with specified parameters.")

    parser.add_argument('--generations', type=int, default=2000, help="Number of generations to run.")
    parser.add_argument('--mutation_probability', type=float, default=20, help="Probability of mutation.")
    parser.add_argument('--keep_elitism', type=int, default=0, help="1 to keep best individual, 0 otherwise.")
    parser.add_argument('--crossover_type', type=str, default="single_point", help="Type of crossover operation.")
    parser.add_argument('--mutation_type', type=str, default="random", help="Type of mutation operation.")
    parser.add_argument('--mutation_percent_genes', type=float, default="1",
                        help="The percentage of genes to be mutated.")
    parser.add_argument('--population', type=int, default=20, help="Starting population size")
    parser.add_argument('--seed', type=int, default=0, help="Seed for images generation")
    parser.add_argument("--cfg_strength", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=20, help="Denoiser steps")
    parser.add_argument("--device", type=str, default="cuda", help="cuda device")
    parser.add_argument("--output", type=str, default="./output/ga_latent/", help="Specifies the output folder")

    args = parser.parse_args()

    return args


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

    _, chad_score = compute_chad_score_from_pil(pil_image)

    return chad_score


def cached_fitness_func(ga_instance, solution, solution_idx):
    sd = ga_instance.sd
    device = ga_instance.device

    solution_copy = solution.copy()  # flatten() is destructive operation
    solution_flattened = solution_copy.flatten()
    solution_reshaped = solution_flattened.reshape(1, 64, 64)

    latent = torch.tensor(solution_reshaped, device=device)
    sd.get_image_from_latent(latent)

    calculate_chad_score()
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
    print(f"Total time taken for Generation #{generation}: {total_time} seconds")
    log_to_file(f"----------------------------------")
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



def main():

    args = parse_args()
    generations = args.generations
    population_size = args.population
    mutation_percent_genes = args.mutation_percent_genes
    mutation_probability = args.mutation_probability
    keep_elitism = args.keep_elitism
    crossover_type = args.crossover_type
    mutation_type = args.mutation_type
    steps = args.steps;

    device = get_device(device=args.device)
    output_directory = args.output
    output_image_directory = join(output_directory, "images/")
    output_feature_directory = join(output_directory, "features/")

    util_clip = UtilClip(device=device)
    util_clip.load_model()

    chad_score_model_path = "input/model/chad_score/chad-score-v1.pth"
    chad_score_predictor = ChadScorePredictor(device=device)
    chad_score_predictor.load_model(chad_score_model_path)

    # make sure the directories are created
    os.makedirs(output_directory, exist_ok=True)
    # Remove the directory and its contents recursively
    shutil.rmtree(output_directory)
    os.makedirs(output_image_directory, exist_ok=True)
    os.makedirs(output_feature_directory, exist_ok=True)


    csv_filename = os.path.join(output_directory, "fitness_data.csv")

    # Write the headers to the CSV file
    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Generation #', 'Population Size', 'Fitness (mean)', 'Fitness (variance)', 'Fitness (best)', 'Fitness array'])

    # TODO: NULL_PROMPT is completely wrong
    NULL_PROMPT = None  # assign later

    # DEVICE = input("Set device: 'cuda:i' or 'cpu'")
    config = ModelPathConfig()

    print(output_directory)
    print(output_image_directory)
    print(output_feature_directory)
    print(csv_filename)

    clip_start_time = time.time()

    # Call the GA loop function with your initialized StableDiffusion model

    parent_selection_type = "tournament"  # "sss", rws, sus, rank, tournament
    # num_parents_mating = int(population_size *.80)
    num_parents_mating = int(population_size * .60)
    # mutation_type = "adaptive" #try adaptive mutation
    # note: uniform is good, two_points"
    crossover_type = "uniform"


    # Load Stable Diffusion
    sd = StableDiffusion(device=device, n_steps=steps)
    sd.quick_initialize().load_autoencoder(config.get_model(SDconfigs.VAE)).load_decoder(config.get_model(SDconfigs.VAE_DECODER))
    sd.model.load_unet(config.get_model(SDconfigs.UNET))

    print("genetic_algorithm_loop: population_size= ", population_size)

    # number of chromozome genes
    num_genes = 64 * 64

    # Create a list to store the random population
    random_population = []

    # Generate and append random arrays to the list
    for _ in range(population_size):
        random_gene = np.random.rand(num_genes)
        random_population.append(random_gene)

    # Initialize the GA
    ga_instance = pygad.GA(initial_population=random_population,
                           num_generations=generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=cached_fitness_func,
                           sol_per_pop=population_size,
                           num_genes=num_genes,
                           # Pygad uses 0-100 range for percentage
                           mutation_percent_genes=mutation_percent_genes,
                           # mutation_probability=mutation_probability,
                           mutation_probability=mutation_probability,
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
                           # fitness_func=calculate_chad_score,
                           # on_parents=on_parents,
                           # on_crossover=on_crossover,
                           on_start=store_generation_images,
                           )
    print(f"Batch Size: {population_size}")
    print((f"Generations: {generations}"))

    # pass custom data to ga_instance
    ga_instance.sd = sd
    ga_instance.device = device

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

