import argparse
import os
import sys
import torch
import configparser
import random

base_dir = os.getcwd()
sys.path.insert(0, base_dir)

from ga.model_clip_text import clip_text_get_prompt_embedding
from model.linear_regression import LinearRegressionModel
from model.util_data_loader import ZipDataLoader

from stable_diffusion import StableDiffusion
from stable_diffusion.utils_backend import get_device
from stable_diffusion.utils_image import to_pil
from stable_diffusion.constants import (IODirectoryTree)
from chad_score.chad_score import ChadScorePredictor


def prompt_embedding_vectors(sd, prompt_array):
    embedded_prompts = clip_text_get_prompt_embedding(ModelConfig=pt, prompts=prompt_array)
    embedded_prompts.to("cpu")
    return embedded_prompts


DEVICE = get_device()
FIXED_SEED = True
SEED = random.randint(0, 2**24)
FIXED_IMAGE_IDX = True

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(os.path.join(base_dir, "config.ini"))
config['BASE']['BASE_DIRECTORY'] = base_dir
config["BASE"].get('base_io_directory')

pt = IODirectoryTree(base_io_directory_prefix = config["BASE"].get('base_io_directory_prefix'), base_directory=base_dir)
pt.create_directory_tree_folders()

# Get CLIP preprocessor
image_features_clip_model, preprocess = clip.load("ViT-L/14", device=DEVICE)

# Load stable diffusion
sd = StableDiffusion(device=DEVICE, n_steps=20)
sd.quick_initialize().load_autoencoder(**pt.autoencoder).load_decoder(**pt.decoder)
sd.model.load_unet(**pt.unet)

# Load chad score
chad_score_model_path = os.path.join('input', 'model', 'chad_score', 'chad-score-v1.pth')
chad_score_predictor = ChadScorePredictor(device=DEVICE)
chad_score_predictor.load_model(chad_score_model_path)

NULL_PROMPT = prompt_embedding_vectors(sd, [""])[0]

def parse_arguments():
    """Command-line arguments for 'classify' command."""
    parser = argparse.ArgumentParser(description="Use gradients to optimize embedding vector.")

    parser.add_argument('--input_path', type=str,
                        default='./input/set_0002.zip',
                        help='Path to input zip')
    parser.add_argument('--model_path', type=str,
                        default='./output/models/prompt_score.pth',
                        help='Path to the model')
    parser.add_argument('--iterations', type=int,
                        default=100,
                        help='How many iterations to perform')
    parser.add_argument('--learning_rate', type=float,
                        default=0.00001,
                        help='Learning rate to use when optimizing')

    return parser.parse_args()

def store_image_from_embeddings(sd, i, prompt_embedding, null_prompt, cfg_strength=9):
    file_dir = os.path.join('output', 'gradient')
    os.makedirs(file_dir, exist_ok=True)
    seed = random.randint(0, 2**24)
    if FIXED_SEED == True:
        seed = 54846
    # prompt_embedding = torch.tensor(prompt_embedding, dtype=torch.float32).to(DEVICE)
    prompt_embedding = prompt_embedding.clone().detach().to(DEVICE)
    prompt_embedding = prompt_embedding.view(1, 77, 768)

    image = sd.generate_images_from_embeddings(
            seed=seed,
            embedded_prompt=prompt_embedding,
            null_prompt=null_prompt,
            uncond_scale=cfg_strength
    )

    prompt_embedding.to("cpu")
    del prompt_embedding

    pil_image = to_pil(image[0])
    filename = os.path.join(file_dir, f'{i}.png')
    pil_image.save(filename)

def main():
    args = parse_arguments()

    zip_file_path = args.input_path
    model_path = args.model_path
    iterations = args.iterations
    learning_rate = args.learning_rate

    inputs = []
    expected_outputs = []

    zip_data_loader = ZipDataLoader()
    loaded_data = zip_data_loader.load(zip_file_path)

    for element in loaded_data:
        image_meta_data = element['image_meta_data']
        embedding_vector = element['embedding_vector']

        embedding_vector = torch.tensor(embedding_vector, dtype=torch.float32, device=DEVICE);
        # Convert the tensor to a flat vector
        flat_embedded_prompts = torch.flatten(embedding_vector)

        with torch.no_grad():
            flat_vector = flat_embedded_prompts.cpu().numpy()

        chad_score = image_meta_data.chad_score

        inputs.append(flat_vector)
        expected_outputs.append(chad_score)

    linear_regression_model = LinearRegressionModel(len(inputs[0]), DEVICE)
    linear_regression_model.load(model_path)

    rand_idx = random.randint(0, len(inputs))
    input = torch.tensor(inputs[rand_idx], dtype=torch.float32, device=DEVICE)
    output = torch.tensor([expected_outputs[rand_idx]], dtype=torch.float32, device=DEVICE)
    gradients = linear_regression_model.compute_gradient(input, output)

    for i in range(iterations):
        input += learning_rate * gradients
        store_image_from_embeddings(sd, i, input, NULL_PROMPT)


if __name__ == '__main__':
    main()
