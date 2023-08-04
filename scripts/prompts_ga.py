import os, sys
base_dir = os.getcwd()
sys.path.insert(0, base_dir)
from chad_score.chad_score import ChadScorePredictor
import torch
from typing import List
import configparser
import hashlib
import json
import math
import numpy as np
import safetensors as st
import random
from os.path import join
# Load chadscore and clip
import clip
import pygad

from stable_diffusion import StableDiffusion
from stable_diffusion.model.clip_text_embedder import CLIPTextEmbedder
from stable_diffusion.model.clip_image_encoder import CLIPImageEncoder

from stable_diffusion.utils_model import *
from stable_diffusion.utils_backend import *
from stable_diffusion.utils_backend import get_device, get_memory_status
from stable_diffusion.utils_model import initialize_latent_diffusion
from stable_diffusion.utils_image import *
from stable_diffusion.constants import IODirectoryTree, create_directory_tree_folders
from stable_diffusion.constants import TOKENIZER_PATH, TEXT_MODEL_PATH
from transformers import CLIPTextModel, CLIPTokenizer

DEVICE = torch.device('cuda:0')
device = DEVICE
image_features_clip_model, preprocess = clip.load("ViT-L/14", device=device)
chad_score_predictor = ChadScorePredictor(device=device)
chad_score_predictor.load_model()

# Variables
SEED = 1337
BATCH_SIZE = 1
POPULATION_SIZE = 12
GEN_IMAGE_N_GENERATIONS = 50
CFG_STRENGTH = 9
N_STEPS = 20
EMBEDDED_PROMPTS_DIR = os.path.abspath(join(base_dir, "./input/embedded_prompts/"))
OUTPUT_DIR = os.path.abspath(
    os.path.join(base_dir, "./output/ga/")
)
IMAGES_DIR = os.path.abspath(join(OUTPUT_DIR, "images/"))
FEATURES_DIR = os.path.abspath(join(OUTPUT_DIR, "features/"))

fitness_cache = {}

# NULL_PROMPT = ""
NULL_PROMPT = torch.tensor(()).to('cuda')
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

# Function to generate prompts
def generate_prompts(prompt_segments, num_prompts=6):
    # Select 6 random segments from the prompt_segments list
    selected_prompts = random.sample(prompt_segments, num_prompts)

    # Add modifiers to the selected prompts
    modifiers = [
        'beautiful', 'gorgeous', 'stunning', 'charming', 'captivating', 'breathtaking',
        'masterpiece', 'exquisite', 'magnificent', 'majestic', 'elegant', 'sublime',
        'ugly', 'hideous', 'grotesque', 'repulsive', 'disgusting', 'revolting',
        'futuristic', 'cyberpunk', 'hi-tech', 'advanced', 'innovative', 'modern',
        'fantasy', 'mythical', 'scifi', 'side scrolling', 'character', 'side scrolling',
        'white background', 'centered', 'full character', 'no background', 'not centered',
        'line drawing', 'sketch', 'black and white', 'colored', 'offset', 'video game']
    prompts_with_modifiers = [f"{modifier} {prompt}" for modifier in modifiers for prompt in selected_prompts]

    # Join the prompts with commas to separate phases
    prompt_phrases = ", ".join(prompts_with_modifiers)

    return prompt_phrases

def embed_and_save_prompts(prompts: list, null_prompt=NULL_PROMPT):
    null_prompt = null_prompt
    prompts = prompts

    clip_text_embedder = CLIPTextEmbedder(device=get_device())
    clip_text_embedder.load_submodels(**pt.embedder_submodels)

    embedded_prompts = []
    for prompt in prompts:
        embeddings = clip_text_embedder.forward(prompt)
        # Flattening tensor and appending
        embedded_prompts.append(embeddings.view(-1))
    embedded_prompts = torch.stack(embedded_prompts)
    torch.save(embedded_prompts, join(EMBEDDED_PROMPTS_DIR, "embedded_prompts.pt"))

    print(
        "Prompts embeddings saved at: ",
        f"{join(EMBEDDED_PROMPTS_DIR, 'embedded_prompts.pt')}",
    )

    get_memory_status()
    clip_text_embedder.to("cpu")
    del clip_text_embedder
    torch.cuda.empty_cache()
    get_memory_status()

    return embedded_prompts, null_prompt

def normalized(a, axis=-1, order=2):
    import numpy as np

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

# Mutation function to add noise to numpy arrays
def mutate(embedded_prompts):
    # Calculate the mean and standard deviation along the last axis (axis=2)
    mean_embedded_prompts = np.mean(embedded_prompts, axis=2)
    std_embedded_prompts = np.std(embedded_prompts, axis=2)

    # Generate noise with the same shape as the embedded_prompts
    noise_shape = (embedded_prompts.shape[0], embedded_prompts.shape[1], 768)
    noise = np.random.normal(loc=mean_embedded_prompts, scale=std_embedded_prompts, size=noise_shape)

    # Generate a random value for the noise multiplier
    random_val = np.random.rand()
    noise_multiplier = min_val + random_val * (0.1 - 0.01)

    # Add noise to the embedded_prompts
    embedding_e = embedded_prompts + noise_multiplier * noise

    return embedding_e

# Crossover function (using single-point crossover)
def crossover(individual1, individual2):
    # Check if the individuals have the same shape
    if individual1.shape != individual2.shape:
        raise ValueError("Individuals must have the same shape for crossover.")

    # Flatten the individuals to make it easier to perform crossover
    flat_individual1 = individual1.flatten()
    flat_individual2 = individual2.flatten()

    # Randomly choose the crossover point
    crossover_point = np.random.randint(0, flat_individual1.shape[0])

    # Perform crossover by swapping genetic material
    offspring1 = np.concatenate((flat_individual1[:crossover_point], flat_individual2[crossover_point:]))
    offspring2 = np.concatenate((flat_individual2[:crossover_point], flat_individual1[crossover_point:]))

    # Reshape the offspring back to the original shape
    offspring1 = offspring1.reshape(individual1.shape)
    offspring2 = offspring2.reshape(individual2.shape)

    return offspring1, offspring2

def generate_images_from_embeddings(embedded_prompts_array, null_prompt):
    # print(embedded_prompts_array.to('cuda0'))
    embedded_prompt = embedded_prompts_array.to('cuda').view(1, 77, 768)
    return sd.generate_images_from_embeddings(
        seed=SEED, embedded_prompt=embedded_prompt, null_prompt=null_prompt)

# Function to calculate the chad score for batch of images
def calculate_chad_score(ga_instance, solution, solution_idx):
    # Convert the numpy array to a PyTorch tensor
    solution_tensor = torch.tensor(solution, dtype=torch.float32)

    # Copy the tensor to CUDA device if 'device' is 'cuda'
    # if device == 'cuda':
    #     solution_reshaped = solution_reshaped.to(device)

    # Generate an image using the solution
    image = generate_images_from_embeddings(solution_tensor, NULL_PROMPT)

    pil_image = to_pil(image[0])  # Convert to (height, width, channels)
    unsqueezed_image = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = image_features_clip_model.encode_image(unsqueezed_image)

        im_emb_arr = normalized(image_features.cpu().detach().numpy() )
        chad_score = chad_score_predictor.get_chad_score(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
        # chad_score = prediction.item()
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
    print("Fitness (mean): ", np.mean(population_fitness_np))
    print("Fitness (variance): ", np.var(population_fitness_np))
    print("Fitness (best): ", np.max(population_fitness_np))

def on_mutation(ga_instance, offspring_mutation):
    print("Performing mutation at generation: ", ga_instance.generations_completed)

def on_generation(ga_instance):
    print("Completed one generation")
    generation = ga_instance.generations_completed
    file_dir = os.path.join(IMAGES_DIR, str(generation))
    os.makedirs(file_dir)
    for i, ind in enumerate(ga_instance.population):
        image = generate_images_from_embeddings(torch.tensor(ind, dtype=torch.float16), NULL_PROMPT)
        pil_image = to_pil(image[0])
        # filename=f"{IMAGES_DIR}/{}/{}.png"
        filename = os.path.join(file_dir, f'{i}.png')
        # pil_image.show()
        pil_image.save(filename)

# Define the GA loop function
def genetic_algorithm_loop(sd, embedded_prompts, null_prompt, generations=10, population_size=POPULATION_SIZE, mutation_rate=0.4, num_parents_mating=2):
    # Move the 'embedded_prompts' tensor to CPU memory
    embedded_prompts_cpu = embedded_prompts.cpu()

    # Reshape the 'embedded_prompts' tensor to a 2D numpy array
    embedded_prompts_array = embedded_prompts_cpu.detach().numpy()
    num_individuals = embedded_prompts_array.shape[0]
    num_genes = embedded_prompts_array.shape[1]
    embedded_prompts_list = embedded_prompts_array.reshape(num_individuals, num_genes).tolist()

    # Initialize the GA
    ga_instance = pygad.GA(num_generations=generations,
                           num_parents_mating=num_parents_mating,
                           # fitness_func=calculate_chad_score,
                           fitness_func=cached_fitness_func,
                           sol_per_pop=population_size,
                           num_genes=num_genes,
                           initial_population=embedded_prompts_list,
                           mutation_percent_genes=mutation_rate*100,
                           #on_start=on_start,
                           on_fitness=on_fitness,
                           #on_parents=on_parents,
                           #on_crossover=on_crossover,
                           on_mutation=on_mutation,
                           on_generation=on_generation,
                           on_stop=on_fitness)

    ga_instance.run()
    return ga_instance.best_solution()

# List of prompt segments
prompt_segments = ['chibi', 'waifu', 'cyborg', 'dragon', 'android', 'nekomimi', 'mecha', 'kitsune', 'AI companion', 'furry detective', 'robot butler', 'futuristic steampunk', 'cybernetic implants', 'anthropomorphic AI', 'mechanical wizard', 'kemonomimi', 'android rebellion', 'magical robot pet', 'intergalactic furball', 'cyberpunk android', 'shapeshifting furry', 'mech pilot', 'furry time traveler']

# Generate 6 random prompts with modifiers (initial population)
PROMPT = generate_prompts(prompt_segments)
PROMPT = PROMPT[:20]

# Load Stable Diffusion
sd = StableDiffusion(device=DEVICE, n_steps=N_STEPS)
sd.quick_initialize().load_autoencoder(**pt.autoencoder).load_decoder(**pt.decoder)
sd.model.load_unet(**pt.unet)

# Generate embeddings for each prompt
embedded_prompts, null_prompt = embed_and_save_prompts(PROMPT)
embedding = embedded_prompts
num_images = embedding.shape[0]

embedded_prompts_cpu = embedded_prompts.cpu()
embedded_prompts_array = embedded_prompts_cpu.detach().numpy()
num_individuals = embedded_prompts_array.shape[0]
num_genes = embedded_prompts_array.shape[1]
embedded_prompts_tensor = torch.tensor(embedded_prompts_array)

# Call the GA loop function with your initialized StableDiffusion model
best_solution = genetic_algorithm_loop(sd, embedded_prompts_tensor, null_prompt, generations=5)
print('best_solution', best_solution)

del preprocess, image_features_clip_model, sd
