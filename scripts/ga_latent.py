import os
import sys
import time

import torch

base_dir = os.getcwd()
sys.path.insert(0, base_dir)

import random
from os.path import join
import shutil
import clip
import pygad
import argparse
import csv
import torch.nn.functional as F

from chad_score.chad_score import ChadScorePredictor
from ga.prompt_generator import generate_prompts
from model.util_clip import UtilClip
from configs.model_config import ModelPathConfig
from stable_diffusion import StableDiffusion, SDconfigs
from stable_diffusion_base_script import StableDiffusionBaseScript
from stable_diffusion.utils_backend import get_autocast, set_seed
# TODO: rename stable_diffusion.utils_backend to /utils/cuda.py
from stable_diffusion.utils_backend import get_device
from stable_diffusion.utils_image import *
from ga.utils import get_next_ga_dir
import ga
from ga.fitness_chad_score import compute_chad_score_from_pil


def parse_args():
    # Add argparse arguments
    parser = argparse.ArgumentParser(description="Run genetic algorithm with specified parameters.")

    parser.add_argument('--generations', type=int, default=100, help="Number of generations to run.")
    parser.add_argument('--mutation_probability', type=float, default=0.001, help="Probability of mutation.")
    parser.add_argument('--keep_elitism', type=int, default=1, help="1 to keep best individual, 0 otherwise.")
    parser.add_argument('--crossover_type', type=str, default="uniform", help="Type of crossover operation.")
    parser.add_argument('--mutation_type', type=str, default="random", help="Type of mutation operation.")
    parser.add_argument('--mutation_percent_genes', type=float, default=0.001,
                        help="The percentage of genes to be mutated.")
    parser.add_argument('--population', type=int, default=80, help="Starting population size")
    parser.add_argument("--steps", type=int, default=20, help="Denoiser steps")
    parser.add_argument("--device", type=str, default="cuda", help="cuda device")
    parser.add_argument("--num_phrases", type=int, default=12, help="number of phrases in the prompt generator")
    parser.add_argument("--cfg_strength", type=float, default=7.5)
    parser.add_argument("--sampler", type=str, default="ddim", help="sampler to use for stable diffusion")
    parser.add_argument("--checkpoint_path", type=str, default="./input/model/sd/v1-5-pruned-emaonly/v1-5-pruned-emaonly.safetensors")
    parser.add_argument("--image_width", type=int, default=512)
    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--output", type=str, default="./output/ga_latent/", help="Specifies the output folder")

    args = parser.parse_args()

    return args



class Txt2Img(StableDiffusionBaseScript):
    """
    ### Text to image class
    """

    @torch.no_grad()
    def generate_images_latent_from_embeddings(self, *,
                                        seed: int = 0,
                                        batch_size: int = 1,
                                        embedded_prompt: torch.Tensor,
                                        null_prompt: torch.Tensor,
                                        h: int = 512, w: int = 512,
                                        uncond_scale: float = 7.5,
                                        low_vram: bool = False,
                                        noise_fn=torch.randn,
                                        temperature: float = 1.0,
                                        ):
        """
        :param seed: the seed to use when generating the images
        :param dest_path: is the path to store the generated images
        :param batch_size: is the number of images to generate in a batch
        :param prompt: is the prompt to generate images with
        :param h: is the height of the image
        :param w: is the width of the image
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param low_vram: whether to limit VRAM usage
        """
        # Number of channels in the image
        c = 4
        # Image to latent space resolution reduction
        f = 8

        if seed == 0:
            seed = time.time_ns() % 2 ** 32

        set_seed(seed)
        # Adjust batch size based on VRAM availability
        if low_vram:
            batch_size = 1

        # AMP auto casting
        autocast = get_autocast()
        with autocast:

            # [Sample in the latent space](../sampler/index.html).
            # `x` will be of shape `[batch_size, c, h / f, w / f]`
            x = self.sampler.sample(cond=embedded_prompt,
                                    shape=[batch_size, c, h // f, w // f],
                                    uncond_scale=uncond_scale,
                                    uncond_cond=null_prompt,
                                    noise_fn=noise_fn,
                                    temperature=temperature)

            return x



def log_to_file(message, output_directory):
    log_path = os.path.join(output_directory, "log.txt")

    with open(log_path, "a") as log_file:
        log_file.write(message + "\n")




def on_fitness(ga_instance, population_fitness):

    csv_filename = ga_instance.csv_filename
    output_directory = ga_instance.output_directory

    population_fitness_np = np.array(population_fitness)
    print("Generation #", ga_instance.generations_completed)
    print("Population Size = ", len(population_fitness_np))
    print("Fitness (mean): ", np.mean(population_fitness_np))
    print("Fitness (variance): ", np.var(population_fitness_np))
    print(f"Fitness (std): {np.sqrt(np.var(population_fitness_np))}")
    print("Fitness (best): ", np.max(population_fitness_np))
    print("fitness array= ", str(population_fitness_np))

    log_to_file(f"Generation #{ga_instance.generations_completed}", output_directory)
    log_to_file(f"Population Size= {len(population_fitness_np)}", output_directory)
    log_to_file(f"Fitness (mean): {np.mean(population_fitness_np)}", output_directory)
    log_to_file(f"Fitness (variance): {np.var(population_fitness_np)}", output_directory)
    log_to_file(f"Fitness (std): {np.sqrt(np.var(population_fitness_np))}", output_directory)
    log_to_file(f"Fitness (best): {np.max(population_fitness_np)}", output_directory)
    log_to_file(f"fitness array= {str(population_fitness_np)}", output_directory)

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
    output_directory = ga_instance.output_directory
    print("Performing mutation at generation: ", ga_instance.generations_completed)
    log_to_file(f"Performing mutation at generation: {ga_instance.generations_completed}", output_directory)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def fitness_func(ga_instance, solution, solution_idx):
    sd = ga_instance.sd
    device = ga_instance.device
    util_clip = ga_instance.util_clip
    chad_score_predictor = ga_instance.chad_score_predictor

    for i in enumerate(solution):
        value = solution[i]
        if value > 1 or value < -1:
            solution[i] = sigmoid(solution[i])

    solution_copy = solution.copy()  # flatten() is destructive operation
    solution_flattened = solution_copy.flatten()
    solution_reshaped = solution_flattened.reshape(1, 4, 64, 64)

    latent = torch.tensor(solution_reshaped, device=device, dtype=torch.float32)
    images = sd.get_image_from_latent(latent)

    # Write the tensor string to the file
    # cleanup
    del latent
    torch.cuda.empty_cache()

    # Map images to `[0, 1]` space and clip
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    # Transpose to `[batch_size, height, width, channels]` and convert to numpy

    images_cpu = images.cpu()

    del images
    torch.cuda.empty_cache()

    images_cpu = images_cpu.permute(0, 2, 3, 1)
    images_cpu = images_cpu.detach().float().numpy()

    image_list = []
    # Save images
    for i, img in enumerate(images_cpu):
        img = Image.fromarray((255. * img).astype(np.uint8))
        image_list.append(img)

    image = image_list[0]

    image_features = util_clip.get_image_features(image)
    # cleanup
    del image
    torch.cuda.empty_cache()

    chad_score = chad_score_predictor.get_chad_score(image_features)
    chad_score_scaled = sigmoid(chad_score)

    # cleanup
    del image_features
    torch.cuda.empty_cache()

    return chad_score_scaled

def store_generation_images(ga_instance):
    device = ga_instance.device
    sd = ga_instance.sd
    output_image_directory = ga_instance.output_image_directory
    output_directory = ga_instance.output_directory

    start_time = time.time()
    generation = ga_instance.generations_completed
    print("Generation #", generation)
    print("Population size: ", len(ga_instance.population))
    file_dir = os.path.join(output_image_directory, str(generation))
    os.makedirs(file_dir)
    for i, gene in enumerate(ga_instance.population):

        solution_copy = gene.copy()
        solution_flattened = solution_copy.flatten()
        solution_reshaped = solution_flattened.reshape(1, 4, 64, 64)

        latent = torch.tensor(solution_reshaped, device=device, dtype=torch.float32)
        image = sd.get_image_from_latent(latent)
        del latent
        torch.cuda.empty_cache()

        pil_image = to_pil(image[0])
        del image
        torch.cuda.empty_cache()

        filename = os.path.join(file_dir, f'g{generation:04}_{i:03}.png')
        pil_image.save(filename)

    end_time = time.time()  # End timing for generation
    total_time = end_time - start_time
    print(f"Total time taken for Generation #{generation}: {total_time} seconds")
    log_to_file(f"----------------------------------", output_directory)
    log_to_file(f"Total time taken for Generation #{generation}: {total_time} seconds", output_directory)

    # Log images per generation
    num_images = len(ga_instance.population)
    print(f"Images generated in Generation #{generation}: {num_images}")
    log_to_file(f"Images generated in Generation #{generation}: {num_images}", output_directory)

    # Log images/sec
    images_per_second = num_images / total_time
    print(f"Images per second in Generation #{generation}: {images_per_second}")
    log_to_file(f"Images per second in Generation #{generation}: {images_per_second}", output_directory)


def main():

    args = parse_args()
    generations = args.generations
    population_size = args.population
    mutation_percent_genes = args.mutation_percent_genes
    mutation_probability = args.mutation_probability
    keep_elitism = args.keep_elitism
    crossover_type = args.crossover_type
    mutation_type = args.mutation_type
    steps = args.steps
    num_phrases = args.num_phrases
    cfg_strength = args.cfg_strength
    sampler = args.sampler
    checkpoint_path = args.checkpoint_path
    image_width = args.image_width
    image_height = args.image_height


    use_randodm_images = False

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


    # Load Stable Diffusion
    sd = StableDiffusion(device=device, n_steps=steps)
    sd.quick_initialize().load_autoencoder(config.get_model(SDconfigs.VAE)).load_decoder(config.get_model(SDconfigs.VAE_DECODER))
    sd.model.load_unet(config.get_model(SDconfigs.UNET))

    print("genetic_algorithm_loop: population_size= ", population_size)

    # Starts the text2img
    txt2img = Txt2Img(
        sampler_name=sampler,
        n_steps=steps,
        force_cpu=False,
        cuda_device=device,
    )
    txt2img.initialize_latent_diffusion(autoencoder=None, clip_text_embedder=None, unet_model=None,
                                        path=checkpoint_path, force_submodels_init=True)

    # number of chromozome genes
    num_genes = 64 * 64 * 4

    # Create a list to store the random population
    random_population = []

    prompt_list = generate_prompts(population_size, num_phrases)

    # Set a seed based on the current time
    seed = int(time.time())
    np.random.seed(seed)
    # Generate and append random arrays to the list
    for i in range(population_size):
        gene = None
        if use_randodm_images:
            this_prompt = prompt_list[i].prompt_str
            # no negative prompts for now
            negative_prompts = []
            un_cond, cond = txt2img.get_text_conditioning(cfg_strength, this_prompt, negative_prompts, 1)
            latent = txt2img.generate_images_latent_from_embeddings(
                batch_size=1,
                embedded_prompt=cond,
                null_prompt=un_cond,
                uncond_scale=cfg_strength,
                seed=seed,
                w=image_width,
                h=image_height
            )
            latent_numpy = latent.cpu().numpy()

            del latent
            torch.cuda.empty_cache()

            gene = latent_numpy.flatten()
        else:
            gene = np.random.rand(num_genes)

        random_population.append(gene)

    # Initialize the GA
    ga_instance = pygad.GA(initial_population=random_population,
                           num_generations=generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
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
                           random_mutation_min_val=5,
                           random_mutation_max_val=10,
                           on_start=store_generation_images,
                           )
    print(f"Batch Size: {population_size}")
    print((f"Generations: {generations}"))

    # pass custom data to ga_instance
    ga_instance.csv_filename = csv_filename
    ga_instance.sd = sd
    ga_instance.device = device
    ga_instance.output_image_directory = output_image_directory
    ga_instance.util_clip = util_clip
    ga_instance.chad_score_predictor = chad_score_predictor
    ga_instance.output_directory = output_directory

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
    log_to_file(f"----------------------------------" , output_directory)
    log_to_file(f"Number of Clip Calculations: {num_clip_calculations} ", output_directory)
    log_to_file(f"Total Time for Clip Calculations: {clip_total_time} seconds", output_directory)
    log_to_file(f"Clip Calculations per Second {clip_calculations_per_second} ", output_directory)



if __name__ == "__main__":
    main()