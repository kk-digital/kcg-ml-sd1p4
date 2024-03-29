import os
import sys
import time
import random
from os.path import join
import shutil
import pygad
import argparse
import csv
import zipfile


base_dir = os.getcwd()
sys.path.insert(0, base_dir)

from chad_score.chad_score import ChadScorePredictor
from model.util_clip import ClipModelHuggingface
from configs.model_config import ModelPathConfig
from stable_diffusion import StableDiffusion, SDconfigs
from stable_diffusion_base_script import StableDiffusionBaseScript
from stable_diffusion.utils_backend import get_autocast, set_seed
# TODO: rename stable_diffusion.utils_backend to /utils/cuda.py
from stable_diffusion.utils_backend import get_device
from stable_diffusion.utils_image import *
from stable_diffusion.model.clip_text_embedder import CLIPTextEmbedder
from ga.similarity_score import get_similarity_score
from ga.chad_score import get_chad_score


def parse_args():
    # Add argparse arguments
    parser = argparse.ArgumentParser(description="Run genetic algorithm with specified parameters.")

    parser.add_argument('--generations', type=int, default=100, help="Number of generations to run.")
    parser.add_argument('--mutation_probability', type=float, default=0.2, help="Probability of mutation.")
    parser.add_argument('--keep_elitism', type=int, default=0, help="1 to keep best individual, 0 otherwise.")
    parser.add_argument('--crossover_type', type=str, default="uniform", help="Type of crossover operation.")
    parser.add_argument('--mutation_type', type=str, default="random", help="Type of mutation operation.")
    parser.add_argument('--mutation_percent_genes', type=float, default=0.1,
                        help="The percentage of genes to be mutated.")
    parser.add_argument('--population', type=int, default=80, help="Starting population size")
    parser.add_argument("--steps", type=int, default=20, help="Denoiser steps")
    parser.add_argument("--device", type=str, default="cuda", help="cuda device")
    parser.add_argument("--num_phrases", type=int, default=12, help="number of phrases in the prompt generator")
    parser.add_argument("--cfg_strength", type=float, default=12)
    parser.add_argument("--sampler", type=str, default="ddim", help="sampler to use for stable diffusion")
    parser.add_argument("--checkpoint_path", type=str, default="./input/model/sd/v1-5-pruned-emaonly/v1-5-pruned-emaonly.safetensors")
    parser.add_argument("--image_width", type=int, default=512)
    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--output", type=str, default="./output/ga_affine_combination_embeddings/", help="Specifies the output folder")
    parser.add_argument("--use_random_images", type=bool, default=False)
    parser.add_argument("--num_prompts", type=int, default=1024)
    parser.add_argument('--prompts_path', type=str, default='/input/prompt-list-civitai/prompt_list_civitai_1000_new.zip')

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

def combine_embeddings(embeddings_array, weight_array, device):

    # empty embedding filled with zeroes
    result_embedding = torch.zeros(1, 77, 768, device=device, dtype=torch.float32)

    # Multiply each tensor by its corresponding float and sum up
    for embedding, weight in zip(embeddings_array, weight_array):
        result_embedding += embedding * weight

    return result_embedding

def combine_embeddings_numpy(embeddings_array_numpy, weight_array, device):

    # empty embedding filled with zeroes
    result_embedding_numpy = np.zeros((1, 77, 768))

    # Multiply each tensor by its corresponding float and sum up
    #for i in range(embeddings_array_numpy):
        #embedding = embeddings_array_numpy[i]
        #weight = weight_array[i]
        #result_embedding_numpy += embedding * weight

    for embedding, weight in zip(embeddings_array_numpy, weight_array):
        result_embedding_numpy += embedding * weight

    result_embedding = torch.from_numpy(result_embedding_numpy)
    result_embedding = result_embedding.to(device)
    result_embedding = result_embedding.to(torch.float32)

    return result_embedding

def combine_clip_embeddings_numpy(embeddings_array_numpy, weight_array, device):

    # empty embedding filled with zeroes
    result_embedding_numpy = np.zeros(768)

    # Multiply each tensor by its corresponding float and sum up
    #for i in range(embeddings_array_numpy):
        #embedding = embeddings_array_numpy[i]
        #weight = weight_array[i]
        #result_embedding_numpy += embedding * weight

    for embedding, weight in zip(embeddings_array_numpy, weight_array):
        result_embedding_numpy += embedding * weight

    result_embedding = torch.from_numpy(result_embedding_numpy)
    result_embedding = result_embedding.to(device)
    result_embedding = result_embedding.to(torch.float32)

    return result_embedding

def generate_image_from_embedding(embeddings_vector, null_prompt, index, seed, output, clip_text_embedder, txt2img, cfg_strength=12,
                          image_width=512, image_height=512):


    latent = txt2img.generate_images_latent_from_embeddings(
        batch_size=1,
        embedded_prompt=embeddings_vector,
        null_prompt=null_prompt,
        uncond_scale=cfg_strength,
        seed=seed,
        w=image_width,
        h=image_height
    )

    images = txt2img.get_image_from_latent(latent)
    generation_dir = output + '/starting_images_from_embeddings/'
    os.makedirs(generation_dir, exist_ok=True)
    image_list, image_hash_list = save_images(images, generation_dir + '/' + str(index + 1) + '.jpg')


def embeddings_chad_score(device, embeddings_vector, negative_embeddings_vector, generation, index, seed, output, chad_score_predictor, clip_text_embedder, txt2img, util_clip, cfg_strength=12,
                          image_width=512, image_height=512):

    latent = txt2img.generate_images_latent_from_embeddings(
        batch_size=1,
        embedded_prompt=embeddings_vector,
        null_prompt=negative_embeddings_vector,
        uncond_scale=cfg_strength,
        seed=seed,
        w=image_width,
        h=image_height
    )

    images = txt2img.get_image_from_latent(latent)
    print('save', ' generation ', generation, ' index : ', index + 1)
    generation_dir = output + '/' + str(generation)
    os.makedirs(generation_dir, exist_ok=True)
    image_list, image_hash_list = save_images(images, generation_dir + '/' + str(index + 1) + '.jpg')

    # Map images to `[0, 1]` space and clip
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images_cpu = images.cpu()
    images_cpu = images_cpu.permute(0, 2, 3, 1)
    images_cpu = images_cpu.detach().float().numpy()

    del latent
    torch.cuda.empty_cache()

    del images
    torch.cuda.empty_cache()

    image_list = []
    # Save images
    for i, img in enumerate(images_cpu):
        img = Image.fromarray((255. * img).astype(np.uint8))
        image_list.append(img)

    image = image_list[0]

    image_features = util_clip.get_image_features(image)
    image_features = image_features.to(torch.float32)

    # get chad score

    chad_score = get_chad_score(chad_score_predictor, image_features)
    chad_score_scaled = torch.sigmoid(chad_score)

    fitness = chad_score_scaled

    # cleanup
    del image_features
    torch.cuda.empty_cache()

    return fitness



def embeddings_similarity_score(device, embeddings_vector, target_features, negative_embeddings_vector, generation, index, seed, output, chad_score_predictor, clip_text_embedder, txt2img, util_clip, cfg_strength=12,
                          image_width=512, image_height=512):

    latent = txt2img.generate_images_latent_from_embeddings(
        batch_size=1,
        embedded_prompt=embeddings_vector,
        null_prompt=negative_embeddings_vector,
        uncond_scale=cfg_strength,
        seed=seed,
        w=image_width,
        h=image_height
    )

    images = txt2img.get_image_from_latent(latent)
    print('save', ' generation ', generation, ' index : ', index + 1)
    generation_dir = output + '/' + str(generation)
    os.makedirs(generation_dir, exist_ok=True)
    image_list, image_hash_list = save_images(images, generation_dir + '/' + str(index + 1) + '.jpg')

    # Map images to `[0, 1]` space and clip
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images_cpu = images.cpu()
    images_cpu = images_cpu.permute(0, 2, 3, 1)
    images_cpu = images_cpu.detach().float().numpy()

    del latent
    torch.cuda.empty_cache()

    del images
    torch.cuda.empty_cache()

    image_list = []
    # Save images
    for i, img in enumerate(images_cpu):
        img = Image.fromarray((255. * img).astype(np.uint8))
        image_list.append(img)

    image = image_list[0]

    image_features = util_clip.get_image_features(image)
    image_features = image_features.to(torch.float32)

    fitness = get_similarity_score(image_features, target_features)

    # cleanup
    del image_features
    torch.cuda.empty_cache()

    return fitness.item()


def embeddings_similarity_and_chad_score(device, embeddings_vector, target_features, negative_embeddings_vector, generation, index, seed, output, chad_score_predictor, clip_text_embedder, txt2img, util_clip, cfg_strength=12,
                          image_width=512, image_height=512):

    latent = txt2img.generate_images_latent_from_embeddings(
        batch_size=1,
        embedded_prompt=embeddings_vector,
        null_prompt=negative_embeddings_vector,
        uncond_scale=cfg_strength,
        seed=seed,
        w=image_width,
        h=image_height
    )

    images = txt2img.get_image_from_latent(latent)
    print('save', ' generation ', generation, ' index : ', index + 1)
    generation_dir = output + '/' + str(generation)
    os.makedirs(generation_dir, exist_ok=True)
    image_list, image_hash_list = save_images(images, generation_dir + '/' + str(index + 1) + '.jpg')

    # Map images to `[0, 1]` space and clip
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images_cpu = images.cpu()
    images_cpu = images_cpu.permute(0, 2, 3, 1)
    images_cpu = images_cpu.detach().float().numpy()

    del latent
    torch.cuda.empty_cache()

    del images
    torch.cuda.empty_cache()


    image_list = []
    # Save images
    for i, img in enumerate(images_cpu):
        img = Image.fromarray((255. * img).astype(np.uint8))
        image_list.append(img)

    image = image_list[0]

    image_features = util_clip.get_image_features(image)
    image_features = image_features.to(torch.float32)

    # get chad score
    chad_score = get_chad_score(chad_score_predictor, image_features)
    chad_score_scaled = torch.sigmoid(chad_score)

    similarity = get_similarity_score(image_features, target_features)

    # if similarity is more than 0.3 fitness is 1.0
    fitness = similarity.item() * 0.5 + chad_score_scaled.item() * 0.5

    # cleanup
    del image_features
    torch.cuda.empty_cache()

    return fitness

def fitness_func(ga_instance, solution, solution_idx):
    sd = ga_instance.sd
    device = ga_instance.device
    util_clip = ga_instance.util_clip
    chad_score_predictor = ga_instance.chad_score_predictor
    clip_text_embedder = ga_instance.clip_text_embedder
    embedded_prompts_array = ga_instance.embedded_prompts_array
    negative_embedded_prompts_array = ga_instance.negative_embedded_prompts_array
    clip_embedded_prompts_array = ga_instance.clip_embedded_prompts_array
    txt2img = ga_instance.txt2img
    weight_array = solution
    output_directory = ga_instance.output_directory
    generation = ga_instance.generations_completed
    cfg_strength = ga_instance.cfg_strength
    image_width = ga_instance.image_width
    image_height = ga_instance.image_height
    fixed_taget_image_features = ga_instance.fixed_taget_image_features

    seed = 6789

    embedding_vector = combine_embeddings_numpy(embedded_prompts_array, weight_array, device)
    clip_embedding_vector = combine_clip_embeddings_numpy(clip_embedded_prompts_array, weight_array, device)

    negative_embedding_vector = combine_embeddings_numpy(negative_embedded_prompts_array, weight_array, device)
    fitness = embeddings_similarity_score(device,
                                            embedding_vector,
                                            fixed_taget_image_features,
                                            negative_embedding_vector,
                                            generation,
                                            solution_idx,
                                            seed,
                                            output_directory,
                                            chad_score_predictor,
                                            clip_text_embedder,
                                            txt2img,
                                            util_clip,
                                            cfg_strength,
                                            image_width,
                                            image_height)



    return fitness

def store_generation_images(ga_instance):

    num_prompts = ga_instance.num_prompts
    mutation_probability = ga_instance.mutation_probability
    mutation_percent_genes = ga_instance.mutation_percent_genes


    for i, ind in enumerate(ga_instance.population):
        # normalize numpy array
        # make sure sum is 1
        #ind /= ind.sum()

        magnitude = np.linalg.norm(ind)
        ind = ind / magnitude

        ga_instance.population[i] = ind

    return 0

def read_prompts_from_zip(zip_file_path, num_prompts):
    # Open the zip file for reading
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Get a list of all file names in the zip archive
        file_list = zip_ref.namelist()
        random.shuffle(file_list)
        # Initialize a list to store loaded arrays
        loaded_arrays = []

        # Iterate over the file list and load the first 100 .npz files
        for file_name in file_list:
            if file_name.endswith('.npz'):
                with zip_ref.open(file_name) as npz_file:
                    npz_data = np.load(npz_file, allow_pickle=True)
                    # Assuming you have a specific array name you want to load from the .npz file
                    loaded_array = npz_data['data']
                    loaded_arrays.append(loaded_array)

            if len(loaded_arrays) >= num_prompts:
                break  # Stop after loading the first 100 .npz files

        return loaded_arrays

def get_target_embeddings_features(util_clip, subject):

    features = util_clip.get_text_features(subject)
    features = features.to(torch.float32)

    features = features.squeeze(0)

    return features

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
    use_random_images = args.use_random_images
    num_prompts = args.num_prompts
    prompts_path = args.prompts_path

    device = get_device(device=args.device)
    output_directory = args.output
    output_image_directory = join(output_directory, "images/")
    output_feature_directory = join(output_directory, "features/")

    seed = 6789

    util_clip = ClipModelHuggingface(device=device)
    util_clip.load_model()

    clip_text_embedder = CLIPTextEmbedder(device=get_device())
    clip_text_embedder.load_submodels()

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
    num_parents_mating = int(population_size * 0.6)
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


    fixed_taget_image_features = get_target_embeddings_features(util_clip, "chibi, anime, waifu, side scrolling")
    # number of chromozome genes
    num_genes = num_prompts

    print("zip path : ", prompts_path)
    prompt_list = read_prompts_from_zip(prompts_path, num_prompts)
    #prompt_list = generate_prompts(num_prompts, num_phrases)

    # embeddings array
    embedded_prompts_array = []
    negative_embedded_prompts_array = []
    clip_embedded_prompts_array = []

    # Get N Embeddings

    idx = 0
    for prompt in prompt_list:
        # get the embedding from positive text prompt
        # prompt_str = prompt.positive_prompt_str

        prompt = prompt.flatten()[0]
        prompt_str = prompt['positive-prompt-str']
        negative_prompt_str = prompt['negative-prompt-str']
        print("positive : ", prompt_str)
        print("negative : ", negative_prompt_str)

        #prompt_str = prompt.positive_prompt_str
        embedded_prompts = clip_text_embedder(prompt_str)
        clip_embeddings = util_clip.get_text_features(prompt_str)
        clip_embeddings = clip_embeddings.squeeze(0)
        print(clip_embeddings.shape)
        negative_embedded_prompts = clip_text_embedder(negative_prompt_str)

        generate_image_from_embedding(embedded_prompts, negative_embedded_prompts, idx, seed, output_directory, clip_text_embedder, txt2img, cfg_strength, image_width, image_height)
        idx = idx + 1

        embedded_prompts_numpy = embedded_prompts.detach().cpu().numpy()
        negative_embedded_prompts_numpy = negative_embedded_prompts.detach().cpu().numpy()
        clip_embedded_prompts_numpy = clip_embeddings.detach().cpu().numpy()

        del embedded_prompts
        del negative_embedded_prompts
        del clip_embeddings

        torch.cuda.empty_cache()

        embedded_prompts_array.append(embedded_prompts_numpy)
        negative_embedded_prompts_array.append(negative_embedded_prompts_numpy)
        clip_embedded_prompts_array.append(clip_embedded_prompts_numpy)

    # Create a list to store the random population
    random_population = []

    for i in range(population_size):
        # Parameters for the Gaussian distribution
        mean = 0  # mean (center) of the distribution
        std_dev = 1  # standard deviation (spread or width) of the distribution
        shape = (num_prompts)  # shape of the resulting array

        # Create the random array
        random_weights = np.random.normal(mean, std_dev, shape)
        random_population.append(random_weights)


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
                           random_mutation_min_val=-(1/num_prompts),
                           random_mutation_max_val=(1/num_prompts),
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
    ga_instance.embedded_prompts_array = embedded_prompts_array
    ga_instance.negative_embedded_prompts_array = negative_embedded_prompts_array
    ga_instance.clip_embedded_prompts_array = clip_embedded_prompts_array
    ga_instance.clip_text_embedder = clip_text_embedder
    ga_instance.txt2img = txt2img
    ga_instance.cfg_strength = cfg_strength
    ga_instance.image_width = image_width
    ga_instance.image_height = image_height
    ga_instance.mutation_probability = mutation_probability
    ga_instance.mutation_percent_genes = mutation_percent_genes
    ga_instance.num_prompts = num_prompts
    ga_instance.fixed_taget_image_features = fixed_taget_image_features

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