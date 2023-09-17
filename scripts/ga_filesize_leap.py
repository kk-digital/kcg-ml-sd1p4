import os
import sys
import time
import random

base_dir = os.getcwd()
sys.path.insert(0, base_dir)

from os.path import join, abspath

import torch
import clip
import argparse

from toolz import pipe
from leap_ec import context
from leap_ec import Individual as leapIndividual
from leap_ec.decoder import Decoder as leapDecoder
from leap_ec.problem import ScalarProblem
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec import util
import leap_ec.ops as ops
import numpy as np

from configs.model_config import ModelPathConfig
from stable_diffusion import StableDiffusion, SDconfigs
from stable_diffusion.utils_backend import get_device
from stable_diffusion.utils_image import to_pil
from ga.utils import get_next_ga_dir
import ga
from ga.fitness_bounding_box_size import size_fitness

# Add argparse arguments
parser = argparse.ArgumentParser(description="Run genetic algorithm with specified parameters.")
parser.add_argument('--generations', type=int, default=2000, help="Number of generations to run.")
args = parser.parse_args()

random.seed()

N_STEPS = 20  # 20, 12
CFG_STRENGTH = 9

FIXED_SEED = False
CONVERT_GREY_SCALE_FOR_SCORING = False

NULL_PROMPT = None

DEVICE = get_device()

config = ModelPathConfig()

EMBEDDED_PROMPTS_DIR = os.path.abspath(join(base_dir, 'input', 'embedded_prompts'))

OUTPUT_DIR = abspath(join(base_dir, 'output', 'ga_filesize_leap'))

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

print(EMBEDDED_PROMPTS_DIR)
print(OUTPUT_DIR)
print(FEATURES_DIR)
print(IMAGES_ROOT_DIR)

# load clip
image_features_clip_model, preprocess = clip.load("ViT-L/14", device=DEVICE)


# Load Stable Diffusion
sd = StableDiffusion(device=DEVICE, n_steps=N_STEPS)
sd.quick_initialize().load_autoencoder(config.get_model(SDconfigs.VAE)).load_decoder(config.get_model(SDconfigs.VAE_DECODER))
sd.model.load_unet(config.get_model(SDconfigs.UNET))

fitness_cache = {}
generation = 0

def calculate_white_background_fitness(solution, generation, solution_idx):
    """Calculates the fitness of a solution and saves the image."""
    
    # Set the random seed for reproducibility
    SEED = random.randint(0, 2 ** 24)
    if FIXED_SEED:
        SEED = 54846

    # Convert the numpy array to a PyTorch tensor
    prompt_embedding = torch.tensor(solution, dtype=torch.float32)
    prompt_embedding = prompt_embedding.view(1, 77, 768).to(DEVICE)

    # Generate the latent representation and the image
    latent = sd.generate_images_latent_from_embeddings(
        seed=SEED,
        embedded_prompt=prompt_embedding,
        null_prompt=NULL_PROMPT,
        uncond_scale=CFG_STRENGTH
    )
    image = sd.get_image_from_latent(latent)
    
    # Cleanup
    del latent
    torch.cuda.empty_cache()
    prompt_embedding.to("cpu")
    del prompt_embedding

    # Convert to PIL image
    pil_image = to_pil(image[0])
    
    # More cleanup
    del image
    torch.cuda.empty_cache()

    # Save the image
    file_dir = os.path.join(IMAGES_ROOT_DIR, str(generation))
    os.makedirs(file_dir, exist_ok=True)
    filename = os.path.join(file_dir, f'g{generation:04}_{solution_idx:03}.png')
    pil_image.save(filename)

    return size_fitness(pil_image)

class Decoder(leapDecoder):
    def __init__(self):
        super().__init__()

    def decode(self, genome, *args, **kwargs):
        return genome.flatten()

    def __repr__(self):
        return type(self).__name__ + "()"

class Problem(ScalarProblem):
    def __init__(self, maximize=True):
        super().__init__(maximize)

    def evaluate(self, phenome):
        return cached_fitness_func(phenome)


class Individual(leapIndividual):
    def __init__(self, genome=[], decoder=None, problem=None, seed=None):
        super().__init__(genome, decoder=decoder, problem=problem)
        if seed is not None:
            random.seed(seed)
        self.seed = seed
        self.fitness = None
        self.individual_seed = random.randint(0, 2 ** 24)

    def set_seed(self, seed):
        self.seed = seed

    def set_individual_seed(self, individual_seed):
        self.individual_seed = individual_seed

    def get_fitness(self):
        return self.problem.evaluate(self.decoder.decode(self.genome))

    def set_genome(self, genome):
        self.genome = genome

    def get_seed(self):
        return self.seed

    def get_individual_seed(self):
        return self.individual_seed

    def get_genome(self):
        return self.genome

    def generate_random_genome(self, gene_count):
        random.seed(self.individual_seed)
        self.genome = [random.random() for _ in range(gene_count)]

    def __str__(self):
        return f"Individual with seed: {self.seed}, individual_seed: {self.individual_seed}, genome: {self.genome}"


# Initialize logger
def log_to_file(message):
    
    log_path = os.path.join(IMAGES_ROOT_DIR, "log.txt")

    with open(log_path, "a") as log_file:
        log_file.write(message + "\n")


def cached_fitness_func(solution):
    if tuple(solution) in fitness_cache:
        print('Returning cached score', fitness_cache[tuple(solution)])
    if tuple(solution) not in fitness_cache:
        # fitness_cache[tuple(solution)] = calculate_chad_score(solution)
        fitness_cache[tuple(solution)] = calculate_white_background_fitness(solution)
        # fitness_cache[tuple(solution)] = calculate_white_border_fitness(solution)
    return fitness_cache[tuple(solution)]
def store_generation_images(population, generation):
    start_time = time.time()  # Start the timer as the function begins

    print("Generation #", generation)
    print("Population size: ", len(population))

    end_time = time.time()  # Get the end time immediately after getting the population size to calculate the total time

    # Calculate and log the total time for this generation
    total_time = end_time - start_time
    log_to_file(f"----------------------------------")
    log_to_file(f"Total time taken for Generation #{generation}: {total_time} seconds")

    # Log the number of images generated in this generation
    num_images = len(population)
    log_to_file(f"Images generated in Generation #{generation}: {num_images}")

    # Calculate and log the generation speed in images per second
    images_per_second = num_images / total_time
    log_to_file(f"Images per second in Generation #{generation}: {images_per_second}")


def on_fitness(generation, population):
    print(f"On_fitness called with generation {generation}")  # Add this line
    population_fitness = [ind.get_fitness() for ind in population]
    population_fitness_np = np.array(population_fitness)
    
    # You can log the fitness values here. For example:
    log_to_file(f"Generation #{generation}")
    log_to_file(f"Population Size= {len(population_fitness_np)}")
    log_to_file(f"Fitness (mean): {np.mean(population_fitness_np)}")
    log_to_file(f"Fitness (variance): {np.var(population_fitness_np)}")
    log_to_file(f"Fitness (best): {np.max(population_fitness_np)}")
    log_to_file(f"Fitness (worst): {np.min(population_fitness_np)}")
    log_to_file(f"Fitness array= {str(population_fitness_np)}")



def prompt_embedding_vectors(sd, prompt_array):
    # Generate embeddings for each prompt
    embedded_prompts = ga.clip_text_get_prompt_embedding(config, prompts=prompt_array)
    # embedded_prompts = []
    # for prompt in prompt_array:
    #     # print(ga.clip_text_get_prompt_embedding(config, prompts=prompt))
    #     random.seed()
    #     embeddings = ga.clip_text_get_prompt_embedding(config, prompts=[prompt]).squeeze().unsqueeze(dim=0)
    #     print(embeddings.shape)
    #     embedded_prompts.append(embeddings)
    #     embeddings.to("cpu")
    # embedded_prompts = torch.stack(embedded_prompts)
    # print(embedded_prompts.shape)
    embedded_prompts.to("cpu")
    return embedded_prompts


def create_individual(genomes):
    counter = [0]
    counter_max = len(genomes)
    def create():
        if (counter[0] >= counter_max):
            raise ValueError("Not enough embedded prompts to create individuals")
        genome = genomes[counter[0]]
        counter[0] += 1
        return np.array(genome)
    return create

# Get embedding of null prompt
NULL_PROMPT = prompt_embedding_vectors(sd, [""])[0]

generations = args.generations
population_size = 20

# generate prompts and get embeddings
prompt_phrase_length = 10  # number of words in prompt
prompts_array = ga.generate_prompts(population_size, prompt_phrase_length)

# get prompt_str array
prompts_str_array = []
for prompt in prompts_array:
    prompt_str = prompt.get_positive_prompt_str()
    prompts_str_array.append(prompt_str)

embedded_prompts = prompt_embedding_vectors(sd, prompt_array=prompts_str_array)

# for embedding in embedded_prompts:
#     print('ueoaueou', embedding[:5])

# print('shapey', embedded_prompts.shape)

embedded_prompts_cpu = embedded_prompts.to("cpu")
embedded_prompts_list = embedded_prompts_cpu.detach().numpy()

# embedded_prompts_list = embedded_prompts_array.reshape(population_size, 77 * 768).tolist()

for embedding in embedded_prompts_list:
    print('embedded_prompts_list', embedding)

# print('shapey2', embedded_prompts_list.shape)

parents = Individual.create_population(len(embedded_prompts_list),
                                       initialize=create_individual(embedded_prompts_list),
                                       decoder=Decoder(),
                                       problem=Problem())

parents = Individual.evaluate_population(parents)
util.print_population(parents, generation=0)

generation_counter = util.inc_generation(context=context)

# Storing for first generation
store_generation_images(parents, 0)

while generation_counter.generation() < generations:
    print("Starting new generation...")
    offspring = pipe(parents,
                     ops.tournament_selection,
                     ops.clone,
                     # mutate_bitflip(expected_num_mutations=1),
                     # mutate_gaussian(std=0.1, hard_bounds=[-1, 1], expected_num_mutations=1),
                     ops.uniform_crossover(p_swap=0.2),
                     ops.evaluate,
                     ops.pool(size=len(parents)))  # accumulate offspring

    parents = offspring

    # Storing images
    store_generation_images(parents, generation_counter.generation() + 1)
    on_fitness(generation_counter.generation(), parents)
    generation_counter()  # increment to the next generation

    util.print_population(parents, context['leap']['generation'])

# log_to_file(f"Batch Size: {population_size}")
# log_to_file(f"Mutation Type: {mutation_type}")
# log_to_file(f"Mutation Rate: {mutation_rate}")
# log_to_file(f"Generations: {generations}")

del preprocess, image_features_clip_model, sd
