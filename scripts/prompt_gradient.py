import argparse
import os
import sys
import torch
import configparser
import random
import clip
import numpy as np
from tabulate import tabulate
from scipy.spatial.distance import cosine

from configs.model_config import ModelPathConfig

base_dir = os.getcwd()
sys.path.insert(0, base_dir)

from ga.model_clip_text import clip_text_get_prompt_embedding
from model.linear_regression import LinearRegressionModel
from model.util_data_loader import ZipDataLoader

from stable_diffusion import StableDiffusion
from stable_diffusion.utils_backend import get_device
from stable_diffusion.utils_image import to_pil
from stable_diffusion.model_paths import (IODirectoryTree, SDconfigs)
from chad_score.chad_score import ChadScorePredictor


def prompt_embedding_vectors(sd, prompt_array):
    embedded_prompts = clip_text_get_prompt_embedding(config, prompts=prompt_array)
    embedded_prompts.to("cpu")
    return embedded_prompts


DEVICE = get_device()
FIXED_SEED = True
SEED = random.randint(0, 2**24)
FIXED_IMAGE_IDX = True

config = ModelPathConfig()

# Get CLIP preprocessor
image_features_clip_model, preprocess = clip.load("ViT-L/14", device=DEVICE)

# Load stable diffusion
sd = StableDiffusion(device=DEVICE, n_steps=20)
sd.quick_initialize().load_autoencoder(config.get_model(SDconfigs.VAE)).load_decoder(config.get_model(SDconfigs.VAE_DECODER))
sd.model.load_unet(config.get_model(SDconfigs.UNET_MODEL))

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
    if FIXED_SEED == True:
        SEED = 54846
    # prompt_embedding = torch.tensor(prompt_embedding, dtype=torch.float32).to(DEVICE)
    prompt_embedding = prompt_embedding.clone().detach().to(DEVICE)
    prompt_embedding = prompt_embedding.view(1, 77, 768)

    image = sd.generate_images_from_embeddings(
            seed=SEED,
            embedded_prompt=prompt_embedding,
            null_prompt=null_prompt,
            uncond_scale=cfg_strength
    )

    prompt_embedding.to("cpu")
    del prompt_embedding

    pil_image = to_pil(image[0])
    filename = os.path.join(file_dir, f'{i}.png')
    pil_image.save(filename)
    return pil_image

# We're taking advantage of the fact that we already created the pil image in
# store_image_from_embeddings
def get_chad_score_from_pil_image(pil_image):
    unsqueezed_image = preprocess(pil_image).unsqueeze(0).to(DEVICE)
    # Get CLIP encoding of model
    with torch.no_grad():
        image_features = image_features_clip_model.encode_image(unsqueezed_image)
        chad_score = chad_score_predictor.get_chad_score(image_features.type(torch.cuda.FloatTensor))
        return chad_score

def report_row(iteration,
               starting_vector,
               updated_vector,
               pil_image,
               starting_model_score,
               model_score):

    cosine_distance = cosine(starting_vector, updated_vector)
    mse = np.mean((starting_vector - updated_vector)**2)
    chad_score = get_chad_score_from_pil_image(pil_image)
    residual = abs(model_score - chad_score)

    row = [iteration,
           "{:.4e}".format(cosine_distance),
           "{:.4e}".format(mse),
           "{:.4f}".format(chad_score),
           "{:.4f}".format(model_score.item()),
           "{:.4f}".format(residual.item())]
    return row

def report_table(rows):
    table_headers = ['Iteration', 'Cosine', 'MSE', 'Chad Score', 'Model Score', 'Residual']

    table = tabulate(rows, headers=table_headers, tablefmt="pretty")

    return table

def report(rows, iterations, learning_rate, starting_model_score, starting_chad_score):
    data_before_table = ""
    data_before_table += f"Seed: {SEED}\n"
    data_before_table += f"Iterations: {iterations}\n"
    data_before_table += f"Learning rate: {learning_rate}\n"
    data_before_table += "\n"

    report = data_before_table + report_table(rows) + "\n"
    return report

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

    image_idx = random.randint(0, len(inputs))
    if FIXED_IMAGE_IDX:
        image_idx = 33
    starting_vector = torch.tensor(inputs[image_idx], dtype=torch.float32, device=DEVICE)
    starting_vector_cpu = starting_vector.cpu()
    starting_vector_np = starting_vector_cpu.numpy()
    starting_model_score = linear_regression_model.model(starting_vector)
    input = starting_vector.clone().detach().to(DEVICE)
    input_cpu = input.cpu()
    input_np = input_cpu.numpy()
    output = torch.tensor([expected_outputs[image_idx]], dtype=torch.float32, device=DEVICE)
    gradients = linear_regression_model.compute_gradient(input, output)
    # Storing original image
    pil_image = store_image_from_embeddings(sd, 0, starting_vector, NULL_PROMPT)
    starting_chad_score = get_chad_score_from_pil_image(pil_image)
    starting_report_row = report_row(0,  # First iteration
                                     starting_vector_np,
                                     input_np,
                                     pil_image,
                                     starting_model_score,
                                     starting_model_score)

    print(report_table([starting_report_row]))

    table_rows = [starting_report_row]
    for i in range(1, iterations + 1):
        input += learning_rate * gradients
        input_cpu = input.cpu()
        input_np = input_cpu.numpy()
        pil_image = store_image_from_embeddings(sd, i, input, NULL_PROMPT)
        model_score = linear_regression_model.model(input)
        row = report_row(i,
                         starting_vector_np,
                         input_np,
                         pil_image,
                         starting_model_score,
                         model_score)
        print(report_table([row]))
        table_rows.append(row)
    report_str = report(table_rows,
                        iterations,
                        learning_rate,
                        starting_model_score,
                        starting_chad_score)
    report_dir = os.path.join('output', 'gradient')
    report_file_path = os.path.join(report_dir, 'report.txt')
    print(report_str)
    with open(report_file_path, "w", encoding="utf-8") as file:
        file.write(report_str)
    print("Report saved at {}".format(report_file_path))


if __name__ == '__main__':
    main()
    del preprocess, image_features_clip_model, sd
