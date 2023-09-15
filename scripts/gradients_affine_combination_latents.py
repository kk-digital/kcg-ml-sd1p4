from PIL import Image

import argparse
import os
import sys
import numpy as np
import torch
import time
import shutil
import torch.nn as nn
import torch.optim as optim
import random
import zipfile
import torchvision.transforms as transforms

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())


from stable_diffusion.model_paths import SD_CHECKPOINT_PATH
from stable_diffusion.utils_backend import get_device
from ga.prompt_generator import generate_prompts
from stable_diffusion.model.clip_text_embedder import CLIPTextEmbedder
from stable_diffusion_base_script import StableDiffusionBaseScript
from stable_diffusion.utils_backend import get_autocast, set_seed
from chad_score.chad_score import ChadScorePredictor
from utility.clip import ClipModel
from stable_diffusion.utils_image import save_images

def parse_arguments():
    """Command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Affine combination of embeddings.")

    parser.add_argument('--output', type=str, default='./output/gradients_affine_combination_latents')
    parser.add_argument('--image_width', type=int, default=512)
    parser.add_argument('--image_height', type=int, default=512)
    parser.add_argument('--cfg_strength', type=float, default=12)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--sampler', type=str, default='ddim')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--checkpoint_path',type=str, default=SD_CHECKPOINT_PATH)
    parser.add_argument('--num_prompts', type=int, default=12)
    parser.add_argument('--num_phrases', type=int, default=12)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_images', type=int, default=1)
    parser.add_argument('--prompts_path', type=str, default='/input/prompt-list-civitai/prompt_list_civitai_1000_new.zip')

    return parser.parse_args()



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


def combine_latents(latents_array, weight_array, device):

    # empty latent filled with zeroes
    result_latents = torch.zeros(1, 4, 64, 64, device=device, dtype=torch.float32)

    # Multiply each tensor by its corresponding float and sum up
    for latent, weight in zip(latents_array, weight_array):
        weighted_latent = latent * weight
        result_latents += weighted_latent

    return result_latents


def get_similarity_score(image_features, target_features):

    image_features_magnitude = torch.norm(image_features)
    target_features_magnitude = torch.norm(target_features)

    image_features = image_features / image_features_magnitude
    target_features = target_features / target_features_magnitude

    image_features = image_features.squeeze(0)

    similarity = torch.dot(image_features, target_features)

    fitness = similarity

    return fitness

def latents_similarity_score(latent, index, output, target_features, device, save_image):

    images = txt2img.get_image_from_latent(latent)
    if save_image:
        image_list, image_hash_list = save_images(images, output + '/image' + str(index + 1) + '.jpg')

    del latent
    torch.cuda.empty_cache()

    # Map images to `[0, 1]` space and clip
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)


    ## Normalize the image tensor
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(-1, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(-1, 1, 1)

    normalized_image_tensor = (images - mean) / std

    # Resize the image to [N, C, 224, 224]
    transform = transforms.Compose([transforms.Resize((224, 224))])
    resized_image_tensor = transform(normalized_image_tensor)

    # Get the CLIP features
    image_features = util_clip.model.encode_image(resized_image_tensor)
    image_features = image_features.squeeze(0)

    image_features = image_features.to(torch.float32)

    # cleanup
    del images
    torch.cuda.empty_cache()

    fitness = get_similarity_score(image_features, target_features)
    print("fitness : ", fitness.item())

    # cleanup
    del image_features
    torch.cuda.empty_cache()

    return fitness

def latents_chad_score(latent, index, output, chad_score_predictor):

    images = txt2img.get_image_from_latent(latent)
    image_list, image_hash_list = save_images(images, output + '/image' + str(index + 1) + '.jpg')

    del latent
    torch.cuda.empty_cache()

    # Map images to `[0, 1]` space and clip
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)


    ## Normalize the image tensor
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(-1, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(-1, 1, 1)

    normalized_image_tensor = (images - mean) / std

    # Resize the image to [N, C, 224, 224]
    transform = transforms.Compose([transforms.Resize((224, 224))])
    resized_image_tensor = transform(normalized_image_tensor)

    # Get the CLIP features
    image_features = util_clip.model.encode_image(resized_image_tensor)
    image_features = image_features.squeeze(0)

    image_features = image_features.to(torch.float32)

    # cleanup
    del images
    torch.cuda.empty_cache()

    chad_score = chad_score_predictor.get_chad_score_tensor(image_features)
    chad_score_scaled = torch.sigmoid(chad_score)

    # cleanup
    del image_features
    torch.cuda.empty_cache()

    return chad_score, chad_score_scaled

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_target_embeddings_features(util_clip, subject):

    features = util_clip.get_text_features(subject)
    features = features.to(torch.float32)

    features = features.squeeze(0)

    return features

def log_to_file(message, output_directory):
    log_path = os.path.join(output_directory, "log.txt")

    with open(log_path, "a") as log_file:
        log_file.write(message + "\n")

if __name__ == "__main__":
    args = parse_arguments()

    output = args.output
    image_width = args.image_width
    image_height = args.image_height
    cfg_strength = args.cfg_strength
    device = get_device(args.device)
    steps = args.steps
    sampler = args.sampler
    checkpoint_path = args.checkpoint_path
    num_prompts = args.num_prompts
    num_phrases = args.num_phrases
    iterations = args.iterations
    learning_rate = args.learning_rate
    num_images = args.num_images
    prompts_path = args.prompts_path

    starting_images_directory = output + '/' + 'starting_images'

    # Seed the random number generator with the current time
    random.seed(time.time())
    seed = 6789

    # make sure the directories are created
    os.makedirs(output, exist_ok=True)

    # Remove the directory and its contents recursively
    shutil.rmtree(output)

    # make sure the directories are created
    os.makedirs(output, exist_ok=True)
    os.makedirs(starting_images_directory, exist_ok=True)

    clip_text_embedder = CLIPTextEmbedder(device=get_device())
    clip_text_embedder.load_submodels()

    # Load default chad model
    # hard coded for now
    chad_score_model_path = "input/model/chad_score/chad-score-v1.pth"
    chad_score_model_name = os.path.basename(chad_score_model_path)
    chad_score_predictor = ChadScorePredictor(device=device)
    chad_score_predictor.load_model(chad_score_model_path)

    # Load the clip model
    util_clip = ClipModel()
    util_clip.load_clip()

    prompt_list = read_prompts_from_zip(prompts_path, num_prompts)
    #prompt_list = generate_prompts(num_images, num_phrases)

    # Starts the text2img
    txt2img = Txt2Img(
        sampler_name=sampler,
        n_steps=steps,
        force_cpu=False,
        cuda_device=device,
    )
    txt2img.initialize_latent_diffusion(autoencoder=None, clip_text_embedder=None, unet_model=None,
                                        path=checkpoint_path, force_submodels_init=True)

    fixed_taget_features = get_target_embeddings_features(util_clip, "chibi, anime, waifu, side scrolling")

    latent_array = []
    index = 0
    # Get N Embeddings
    for prompt in prompt_list:
        index = index + 1
        # get the embedding from positive text prompt
        # prompt_str = prompt.positive_prompt_str
        prompt = prompt.flatten()[0]

        prompt_str = prompt['positive-prompt-str']
        negative_prompt_str = prompt['negative-prompt-str']

        print("positive : " + prompt_str)
        print("negative : " + negative_prompt_str)

        embedded_prompts = clip_text_embedder(prompt_str)
        negative_embedded_prompts = clip_text_embedder(negative_prompt_str)

        latent = txt2img.generate_images_latent_from_embeddings(
            batch_size=1,
            embedded_prompt=embedded_prompts,
            null_prompt=negative_embedded_prompts,
            uncond_scale=cfg_strength,
            seed=seed,
            w=image_width,
            h=image_height
        )

        latent = latent.clone()

        images = txt2img.get_image_from_latent(latent)
        image_list, image_hash_list = save_images(images, starting_images_directory + '/image' + str(index + 1) + '.jpg')

        latent_array.append(latent)

    # Parameters for the Gaussian distribution
    mean = 0  # mean (center) of the distribution
    std_dev = 1  # standard deviation (spread or width) of the distribution
    shape = (num_prompts)  # shape of the resulting array

    # Create the random weights
    weight_array = np.random.normal(mean, std_dev, shape)

    # normalize the array
    magnitude = np.linalg.norm(weight_array)
    weight_array = weight_array / magnitude

    # convert to tensor
    weight_array = torch.tensor(weight_array, device=device, dtype=torch.float32, requires_grad=True)

    optimizer = optim.AdamW([weight_array], lr=learning_rate, weight_decay=0.01)
    mse_loss = nn.MSELoss(reduction='sum')

    target = torch.tensor([1.0], device=device, dtype=torch.float32, requires_grad=True)

    start_time = time.time()

    fixed_taget_features = get_target_embeddings_features(util_clip, "chibi, anime, waifu, side scrolling")

    for i in range(0, iterations):
        # Zero the gradients

        fixed_taget = fixed_taget_features.detach().clone()

        save_image = True

        combined_latent = combine_latents(latent_array, weight_array, device)

        #chad_score, chad_score_scaled = latents_chad_score(combined_latent, i, output, chad_score_predictor)
        fitness = latents_similarity_score(combined_latent, i, output, fixed_taget, device, save_image)

        input = fitness
        loss = mse_loss(input, target)

        print(f'Iteration #{i + 1}, loss {loss}')
        log_to_file(f'Iteration #{i + 1}, loss {loss}', output)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Time elapsed: {elapsed_time:.4f} seconds")
