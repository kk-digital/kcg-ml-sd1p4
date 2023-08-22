import os
import sys
import time
import random

base_dir = os.getcwd()
sys.path.insert(0, base_dir)

from os.path import join

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

from chad_score.chad_score import ChadScorePredictor
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

# load clip
image_features_clip_model, preprocess = clip.load("ViT-L/14", device=DEVICE)

# load chad score
chad_score_model_path = os.path.join('input', 'model', 'chad_score', 'chad-score-v1.pth')
chad_score_predictor = ChadScorePredictor(device=DEVICE)
chad_score_predictor.load_model(chad_score_model_path)

# Load Stable Diffusion
sd = StableDiffusion(device=DEVICE, n_steps=N_STEPS)
sd.quick_initialize().load_autoencoder(config.get_model(SDconfigs.VAE)).load_decoder(config.get_model(SDconfigs.VAE_DECODER))
sd.model.load_unet(config.get_model(SDconfigs.UNET))

fitness_cache = {}

def calculate_size_fitness(solution):
    # set seed
    SEED = random.randint(0, 2 ** 24)
    if FIXED_SEED == True:
        SEED = 54846

    # Convert the numpy array to a PyTorch tensor
    prompt_embedding = torch.tensor(solution, dtype=torch.float32)
    prompt_embedding = prompt_embedding.view(1, 77, 768).to(DEVICE)

    # NOTE: Is using NoGrad internally
    # NOTE: Is using autocast internally
    latent = sd.generate_images_latent_from_embeddings(
        seed=SEED,
        embedded_prompt=prompt_embedding,
        null_prompt=NULL_PROMPT,
        uncond_scale=CFG_STRENGTH
    )

    image = sd.get_image_from_latent(latent)

    # move back to cpu
    prompt_embedding.to("cpu")
    del prompt_embedding

    pil_image = to_pil(image[0])  # Convert to (height, width, channels)

    # convert to grey scale
    if CONVERT_GREY_SCALE_FOR_SCORING == True:
        pil_image = pil_image.convert("L")
        pil_image = pil_image.convert("RGB")
    return size_fitness(pil_image)





def count_contiguous_high_intensity_pixels(image, start_row, start_col, threshold):
    width, height = image.size
    visited = set()
    queue = [(start_row, start_col)]
    high_intensity_pixel_count = 0

    while queue:
        row, col = queue.pop(0)
        if (row, col) in visited or not (0 <= row < height and 0 <= col < width):
            continue

        visited.add((row, col))
        pixel = image.getpixel((col, row))  # Get the pixel value at (x, y)
        intensity = sum(pixel) / 3  # Calculate average intensity (assuming RGB image)
        if intensity > threshold:
            high_intensity_pixel_count += 1
            neighbors = [
                (row + 1, col),
                (row - 1, col),
                (row, col + 1),
                (row, col - 1)
            ]
            queue.extend(neighbors)

    return high_intensity_pixel_count

def find_largest_contiguous_high_intensity_area(image, threshold):
    width, height = image.size
    corner_high_intensity_counts = [
        count_contiguous_high_intensity_pixels(image, 0, 0, threshold),
        count_contiguous_high_intensity_pixels(image, 0, width - 1, threshold),
        count_contiguous_high_intensity_pixels(image, height - 1, 0, threshold),
        count_contiguous_high_intensity_pixels(image, height - 1, width - 1, threshold)
    ]
    max_high_intensity_count = max(corner_high_intensity_counts)
    winning_corner = corner_high_intensity_counts.index(max_high_intensity_count)
    proportion = max_high_intensity_count / (width * height)

    return proportion

def calculate_distance_from_white(pixel_value, max_distance):
    return np.linalg.norm(pixel_value - 255) / max_distance

def sigmoid(x, alpha=10):
    return 1 / (1 + np.exp(-alpha * (x - 0.5)))

# def calculate_white_border_score(image):
#     # weights = sigmoid(np.linspace(1.0, -1.0, 512 // 2 + 1))
#     weights = sigmoid(np.linspace(1.0, 0.0, 512 // 2 + 1))
#     # weights = np.linspace(1.0, -1.0, 512 // 2 + 1)
#     gray_image = image.convert("L")
#     width, height = gray_image.size
#     max_distance = np.sqrt((255 ** 2) * 2)  # Maximum distance for a pixel value from pure white

#     distance_matrix = np.zeros((height, width), dtype=np.float32)

#     for i in range(min(height, width) // 2 + 1):
#         # Traverse top
#         for j in range(i, width - i):
#             pixel_value = gray_image.getpixel((j, i))
#             distance = calculate_distance_from_white(pixel_value, max_distance)
#             distance_matrix[i, j] = distance * weights[i]

#         # Traverse right
#         for j in range(i + 1, height - i):
#             pixel_value = gray_image.getpixel((width - i - 1, j))
#             distance = calculate_distance_from_white(pixel_value, max_distance)
#             distance_matrix[j, width - i - 1] = distance * weights[i]

#         # Traverse bottom
#         for j in range(i, width - i - 1)[::-1]:
#             pixel_value = gray_image.getpixel((j, height - i - 1))
#             distance = calculate_distance_from_white(pixel_value, max_distance)
#             distance_matrix[height - i - 1, j] = distance * weights[i]

#         # Traverse left
#         for j in range(i + 1, height - i - 1)[::-1]:
#             pixel_value = gray_image.getpixel((i, j))
#             distance = calculate_distance_from_white(pixel_value, max_distance)
#             distance_matrix[j, i] = distance * weights[i]

#     return 1 - np.mean(distance_matrix != 0.0)

# def calculate_white_border_score(image):
#     # Convert image to grayscale
#     gray_image = image.convert("L")

#     # Get image dimensions
#     width, height = image.size

#     # Define the central area bounds
#     center_x1 = (width - 64) // 2
#     center_y1 = (height - 64) // 2
#     center_x2 = center_x1 + 64
#     center_y2 = center_y1 + 64

#     # Initialize variables for sum and count of pixel distances
#     sum_distances = 0
#     count = 0

#     # Iterate through each pixel in the image
#     for y in range(height):
#         for x in range(width):
#             # Check if the pixel is outside the central area
#             if center_x1 <= x < center_x2 and center_y1 <= y < center_y2:
#                 continue

#             # Calculate the distance of the pixel from white (255)
#             pixel_value = gray_image.getpixel((x, y))
#             distance = abs(pixel_value - 255)

#             # Add the distance to the sum and increment the count
#             sum_distances += distance
#             count += 1

#     # Calculate the mean of pixel distances
#     mean_distance = sum_distances / count

#     return 1 - mean_distance

def calculate_white_border_score(image):
    # Convert image to grayscale
    gray_image = image.convert("L")

    # Get image dimensions
    width, height = image.size

    # Define the central area bounds
    center_x1 = (width - 64) // 2
    center_y1 = (height - 64) // 2
    center_x2 = center_x1 + 64
    center_y2 = center_y1 + 64

    # Initialize variable for sum of pixel distances
    sum_distances = 0

    # Iterate through each pixel in the image
    for y in range(height):
        for x in range(width):
            # Check if the pixel is outside the central area
            if center_x1 <= x < center_x2 and center_y1 <= y < center_y2:
                continue

            # Calculate the average distance from white for the pixel and its neighbors
            neighbor_distances = []
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue

                    neighbor_x = x + dx
                    neighbor_y = y + dy

                    # Check if the neighbor pixel is within the image bounds
                    if 0 <= neighbor_x < width and 0 <= neighbor_y < height:
                        pixel_value = gray_image.getpixel((neighbor_x, neighbor_y))
                        distance = abs(pixel_value - 255)
                        neighbor_distances.append(distance)

            # Calculate the average distance from white for the pixel and its neighbors
            avg_distance = sum(neighbor_distances) / len(neighbor_distances)
            sum_distances += avg_distance

    # Calculate the mean of pixel distances
    mean_distance = sum_distances / (width * height)

    return mean_distance

def calculate_white_background_fitness(solution):
    # set seed
    SEED = random.randint(0, 2 ** 24)
    if FIXED_SEED == True:
        SEED = 54846

    # Convert the numpy array to a PyTorch tensor
    prompt_embedding = torch.tensor(solution, dtype=torch.float32)
    prompt_embedding = prompt_embedding.view(1, 77, 768).to(DEVICE)

    # NOTE: Is using NoGrad internally
    # NOTE: Is using autocast internally
    latent = sd.generate_images_latent_from_embeddings(
        seed=SEED,
        embedded_prompt=prompt_embedding,
        null_prompt=NULL_PROMPT,
        uncond_scale=CFG_STRENGTH
    )

    image = sd.get_image_from_latent(latent)

    # move back to cpu
    prompt_embedding.to("cpu")
    del prompt_embedding

    pil_image = to_pil(image[0])  # Convert to (height, width, channels)

    return calculate_white_border_score(pil_image)

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
        # return calculate_size_fitness(phenome)


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
    
    log_path = os.path.join(OUTPUT_DIR, "log.txt")

    with open(log_path, "a") as log_file:
        log_file.write(message + "\n")


# Function to calculate the chad score for batch of images
def calculate_chad_score(solution):
    # set seed
    SEED = random.randint(0, 2 ** 24)
    if FIXED_SEED == True:
        SEED = 54846

    # Convert the numpy array to a PyTorch tensor
    prompt_embedding = torch.tensor(solution, dtype=torch.float32)
    prompt_embedding = prompt_embedding.view(1, 77, 768).to(DEVICE)

    # NOTE: Is using NoGrad internally
    # NOTE: Is using autocast internally
    latent = sd.generate_images_latent_from_embeddings(
        seed=SEED,
        embedded_prompt=prompt_embedding,
        null_prompt=NULL_PROMPT,
        uncond_scale=CFG_STRENGTH
    )

    image = sd.get_image_from_latent(latent)

    # move back to cpu
    prompt_embedding.to("cpu")
    del prompt_embedding

    pil_image = to_pil(image[0])  # Convert to (height, width, channels)

    # convert to grey scale
    if CONVERT_GREY_SCALE_FOR_SCORING == True:
        pil_image = pil_image.convert("L")
        pil_image = pil_image.convert("RGB")

    unsqueezed_image = preprocess(pil_image).unsqueeze(0).to(DEVICE)
    # get clip encoding of model
    with torch.no_grad():
        image_features = image_features_clip_model.encode_image(unsqueezed_image)
        chad_score = chad_score_predictor.get_chad_score(image_features.type(torch.cuda.FloatTensor))
        return chad_score


def cached_fitness_func(solution):
    if tuple(solution) in fitness_cache:
        print('Returning cached score', fitness_cache[tuple(solution)])
    if tuple(solution) not in fitness_cache:
        # fitness_cache[tuple(solution)] = calculate_chad_score(solution)
        # fitness_cache[tuple(solution)] = calculate_size_fitness(solution)
        fitness_cache[tuple(solution)] = calculate_white_background_fitness(solution)
        # fitness_cache[tuple(solution)] = calculate_white_border_fitness(solution)
    return fitness_cache[tuple(solution)]


def store_generation_images(population, generation):
    start_time = time.time()
    print("Generation #", generation)
    print("Population size: ", len(population))
    file_dir = os.path.join(IMAGES_DIR, str(generation))
    os.makedirs(file_dir)
    for i, ind in enumerate(population):
        SEED = random.randint(0, 2 ** 24)
        if FIXED_SEED == True:
            SEED = 54846
        prompt_embedding = torch.tensor(ind.get_genome(), dtype=torch.float32).to(DEVICE)
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
    log_to_file(f"Total time taken for Generation #{generation}: {total_time} seconds")
    
    # Log images per generation
    num_images = len(population)
    
    log_to_file(f"Images generated in Generation #{generation}: {num_images}")
    
    # Log images/sec
    images_per_second = num_images / total_time
    log_to_file(f"Images per second in Generation #{generation}: {images_per_second}")


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
    prompt_str = prompt.get_prompt_str()
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

    generation_counter()  # increment to the next generation

    util.print_population(parents, context['leap']['generation'])

# log_to_file(f"Batch Size: {population_size}")
# log_to_file(f"Mutation Type: {mutation_type}")
# log_to_file(f"Mutation Rate: {mutation_rate}")
# log_to_file(f"Generations: {generations}")

del preprocess, image_features_clip_model, sd
