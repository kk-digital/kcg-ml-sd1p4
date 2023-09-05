from PIL import Image

import argparse
import os
import sys
import numpy as np
import torch
import time
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
from model.util_clip import UtilClip
from stable_diffusion.utils_image import save_images

def parse_arguments():
    """Command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Affine combination of embeddings.")

    parser.add_argument('--output', type=str, default='./output')
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
    parser.add_argument('--prompts_path', type=str, default='/input/prompt-list-civitai/prompt_list_civitai_10k_512_phrases.zip')

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


# Now, loaded_arrays contains the loaded NumPy arrays from the first 100 .npz files

def combine_embeddings(embeddings_array, weight_array, device):

    # empty embedding filled with zeroes
    result_embedding = torch.zeros(1, 77, 768, device=device, dtype=torch.float32)

    # Multiply each tensor by its corresponding float and sum up
    for embedding, weight in zip(embeddings_array, weight_array):
        result_embedding += embedding * weight

    return result_embedding


def embeddings_chad_score(embeddings_vector, seed, index, output, chad_score_predictor):

    null_prompt = clip_text_embedder('')

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
    image_list, image_hash_list = save_images(images, output + '/image' + str(index + 1) + '.jpg')

    del latent
    torch.cuda.empty_cache()

    # Map images to `[0, 1]` space and clip
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)


    #image_features = util_clip.get_image_features(images)

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

    #image_input = util_clip.preprocess(images=images, return_tensors="pt")
    #image_input = image_input.unsqueeze(0)

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

    # Seed the random number generator with the current time
    random.seed(time.time())
    seed = 6789

    clip_text_embedder = CLIPTextEmbedder(device=get_device())
    clip_text_embedder.load_submodels()

    # Load default chad model
    # hard coded for now
    chad_score_model_path = "input/model/chad_score/chad-score-v1.pth"
    chad_score_model_name = os.path.basename(chad_score_model_path)
    chad_score_predictor = ChadScorePredictor(device=device)
    chad_score_predictor.load_model(chad_score_model_path)

    # Load the clip model
    util_clip = UtilClip(device=device)
    util_clip.load_model()

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


    embedded_prompts_array = []
    # Get N Embeddings
    for prompt in prompt_list:
        # get the embedding from positive text prompt
        # prompt_str = prompt.positive_prompt_str
        prompt = prompt.flatten()[0]
        prompt_str = prompt['positive-prompt-str']
        embedded_prompts = clip_text_embedder(prompt_str)

        embedded_prompts_array.append(embedded_prompts)

    # array of  weights
    weight_array = np.full(num_prompts, 1.0 / num_prompts)
    weight_array = torch.tensor(weight_array, device=device, dtype=torch.float32, requires_grad=True)
    # Combinate into one Embedding

    optimizer = optim.Adam([weight_array], lr=learning_rate)
    mse_loss = nn.MSELoss(reduction='sum')

    target = torch.tensor([1.0], device=device, dtype=torch.float32, requires_grad=True)

    start_time = time.time()
    for i in range(0, iterations):
        # Zero the gradients
        optimizer.zero_grad()

        embedding_vector = combine_embeddings(embedded_prompts_array, weight_array, device)

        chad_score, chad_score_scaled = embeddings_chad_score(embedding_vector, seed, i, output, chad_score_predictor)

        input = chad_score_scaled
        loss = mse_loss(input, target)

        loss.backward()
        optimizer.step()
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Time elapsed: {elapsed_time:.4f} seconds")
