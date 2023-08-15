import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
import warnings
from os.path import join
from random import randrange
from typing import List

import cv2
import numpy as np
import torch

base_dir = "./"
sys.path.insert(0, base_dir)

from chad_score.chad_score import ChadScorePredictor
from stable_diffusion.utils_backend import get_device, get_memory_status
from stable_diffusion.utils_image import to_pil

warnings.filterwarnings("ignore", category=UserWarning)
from stable_diffusion.model.clip_text_embedder import CLIPTextEmbedder
from stable_diffusion.model.clip_image_encoder import CLIPImageEncoder
from stable_diffusion import StableDiffusion
from stable_diffusion.model_paths import IODirectoryTree, SDconfigs
from configs.model_config import ModelPathConfig

EMBEDDED_PROMPTS_DIR = os.path.abspath("./input/embedded_prompts/")
OUTPUT_DIR = "./output/disturbing_embeddings/"
FEATURES_DIR = join(OUTPUT_DIR, "features/")
IMAGES_DIR = join(OUTPUT_DIR, "images/")
# SCORER_CHECKPOINT_PATH = os.path.abspath("./input/model/aesthetic_scorer/sac+logos+ava1-l14-linearMSE.pth")
SCORER_CHECKPOINT_PATH = os.path.abspath("./input/model/aesthetic_scorer/chadscorer.pth")

# DEVICE = input("Set device: 'cuda:i' or 'cpu'")

prompt_list = ['chibi', 'waifu', 'scifi', 'side scrolling', 'character', 'side scrolling',
               'white background', 'centered', 'full character', 'no background',
               'not centered', 'line drawing', 'sketch', 'black and white',
               'colored', 'offset', 'video game', 'exotic', 'sureal', 'miltech', 'fantasy',
               'frank frazetta', 'terraria', 'final fantasy', 'cortex command',
               'Dog', 'Cat', 'Space Ship', 'Airplane', 'Mech', 'Tank', 'Bicycle',
               'Book', 'Chair', 'Table', 'Cup', 'Car', 'Tree', 'Flower', 'Mountain',
               'Smartphone', 'Guitar', 'Sunflower', 'Laptop', 'Coffee Mug', 'water color expressionist',
               'david mckean', 'jock', 'esad ribic', 'chris bachalo', 'expressionism', 'Jackson Pollock',
               'Alex Kanevskyg', 'Francis Bacon', 'Trash Polka', 'abstract realism', 'andrew salgado',
               'alla prima technique',
               'alla prima', 'expressionist alla prima', 'expressionist alla prima technique']

parser = argparse.ArgumentParser("Embed prompts using CLIP")

parser.add_argument(
    "--cfg_strength",
    type=int,
    default=12,
    help="Configuration strength. Defaults to 12.",
)
parser.add_argument(
    "--embedded_prompts_dir",
    type=str,
    default=EMBEDDED_PROMPTS_DIR,
    help="The path to the directory containing the embedded prompts tensors. Defaults to a constant EMBEDDED_PROMPTS_DIR, which is expected to be './input/embedded_prompts/'",
)
parser.add_argument(
    "--num_iterations",
    type=int,
    default=8,
    help="The number of iterations to batch-generate images. Defaults to 8.",
)

parser.add_argument(
    "--batch_size",
    type=str,
    default=1,
    help="The number of images to generate per batch. Defaults to 1.",
)

parser.add_argument(
    "--seed",
    type=str,
    default='',
    help="The noise seed used to generate the images. Defaults to random int 0 to 2^24",
)
parser.add_argument(
    "--noise_multiplier",
    type=float,
    default=0.01,
    help="The multiplier for the amount of noise used to disturb the prompt embedding. Defaults to 0.008.",
)
parser.add_argument(
    "--cuda_device",
    type=str,
    default=get_device(),
    help="The cuda device to use. Defaults to 'cuda:0'.",
)
parser.add_argument(
    "--clear_output_dir",
    type=bool,
    default=False,
    help="Avoid. If True, the output directory will be cleared before generating images. Defaults to False.",
)

args = parser.parse_args()

NULL_PROMPT = ""
NUM_ITERATIONS = args.num_iterations

if args.seed == '':
    SEED = randrange(0, 2 ** 24)
else:
    SEED = int(args.seed)

NOISE_MULTIPLIER = args.noise_multiplier
DEVICE = args.cuda_device
BATCH_SIZE = args.batch_size
CLEAR_OUTPUT_DIR = args.clear_output_dir
os.makedirs(EMBEDDED_PROMPTS_DIR, exist_ok=True)

config = ModelPathConfig()
pt = IODirectoryTree(config)

try:
    shutil.rmtree(OUTPUT_DIR)
except Exception as e:
    print(e, "\n", "Creating the paths...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
else:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)


def init_stable_diffusion(device, path_tree: ModelPathConfig, sampler_name="ddim", n_steps=20, ddim_eta=0.0):
    device = get_device(device)

    stable_diffusion = StableDiffusion(
        device=device, sampler_name=sampler_name, n_steps=n_steps, ddim_eta=ddim_eta
    )

    stable_diffusion.quick_initialize()
    stable_diffusion.model.load_unet(path_tree.get_model(SDconfigs.UNET))
    autoencoder = stable_diffusion.model.load_autoencoder(path_tree.get_model(SDconfigs.VAE))
    autoencoder.load_decoder(path_tree.get_model(SDconfigs.VAE_DECODER))

    return stable_diffusion


def generate_prompt():
    # Your mandatory phrases
    mandatory_phrases = ['no background', 'white background', 'centered']

    # Add the mandatory phrases to the prompt multiple times (e.g., 2 times)
    mandatory_prompts = mandatory_phrases * 2

    # Select remaining items randomly from the prompt_list
    remaining_items = 9  # 15 - 3 mandatory phrases * 2 occurrences each = 9
    random_prompts = random.sample(set(prompt_list) - set(mandatory_phrases), remaining_items)

    # Join all selected prompts and mandatory prompts into a single string, separated by commas
    prompt = ', '.join(mandatory_prompts + random_prompts)

    print(f"Generated prompt: {prompt}")
    return prompt


def embed_and_save_prompts(clip_text_embedder, prompt: str, i: int, null_prompt=NULL_PROMPT):
    clip_text_embedder = CLIPTextEmbedder(device=get_device())
    clip_text_embedder.load_submodels()

    null_cond = clip_text_embedder(null_prompt)

    torch.save(null_cond, join(EMBEDDED_PROMPTS_DIR, f"null_cond.pt"))
    print(
        "Null prompt embedding saved at: ",
        f"{join(EMBEDDED_PROMPTS_DIR, f'null_cond.pt')}",
    )

    embedded_prompt_tensor = clip_text_embedder(prompt)
    embedded_prompt = embedded_prompt_tensor.cpu()

    embedded_prompt_tensor.detach()
    del embedded_prompt_tensor
    torch.cuda.empty_cache()

    torch.save(embedded_prompt, join(EMBEDDED_PROMPTS_DIR, f"embedded_prompt_{i}.pt"))

    print(
        "Prompts embeddings saved at: ",
        f"{join(EMBEDDED_PROMPTS_DIR, f'embedded_prompt_{i}.pt')}",
    )

    get_memory_status(DEVICE)
    # clip_text_embedder.to("cpu")
    torch.cuda.empty_cache()
    get_memory_status(DEVICE)
    return embedded_prompt, null_cond


def generate_images_from_disturbed_embeddings(
        sd: StableDiffusion,
        clip_text_embedder: CLIPTextEmbedder,
        prompts: List[str],
        # null_prompt: torch.Tensor,
        device=DEVICE,
        seed=SEED,
        num_iterations=NUM_ITERATIONS,
        noise_multiplier=NOISE_MULTIPLIER,
        batch_size=BATCH_SIZE
):
    null_prompt = clip_text_embedder(NULL_PROMPT).to(device)

    for i in range(0, num_iterations):
        embedded_prompt = clip_text_embedder(prompts[i]).to(device)
        print(embedded_prompt.shape)
        dist = torch.distributions.normal.Normal(
            loc=embedded_prompt.mean(dim=2), scale=embedded_prompt.std(dim=2)
        )
        j = num_iterations - i

        noise_i = (
            dist.sample(sample_shape=torch.Size([768])).permute(1, 0, 2).permute(0, 2, 1)
        ).to(device)
        noise_j = (
            dist.sample(sample_shape=torch.Size([768])).permute(1, 0, 2).permute(0, 2, 1)
        ).to(device)
        embedding_e = embedded_prompt + ((i * noise_multiplier) * noise_i + (j * noise_multiplier) * noise_j) / (
                2 * num_iterations)

        latent = sd.generate_images_latent_from_embeddings(
            seed=seed,
            embedded_prompt=embedding_e,
            null_prompt=null_prompt,
            batch_size=batch_size
        )

        image_e = sd.get_image_from_latent(latent)

        embedding_e_cpu = embedding_e.cpu()

        embedding_e.detach()
        del embedding_e
        torch.cuda.empty_cache()

        yield (image_e, embedding_e_cpu, i)


def get_bounding_box_details(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    ret, thresh = cv2.threshold(inverted, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
    areas = [cv2.contourArea(cnt) for cnt in contours]
    largest_bounding_box = bounding_boxes[np.argmax(areas)]
    return largest_bounding_box


def get_bounding_box_center_offset(largest_bounding_box, image_size):
    box_x, box_y, box_w, box_h = largest_bounding_box
    center_x = (box_x + box_w / 2) / image_size[1]
    center_y = (box_y + box_h / 2) / image_size[0]
    center_offset_x = center_x - 0.5
    center_offset_y = center_y - 0.5
    box_w /= image_size[1]
    box_h /= image_size[0]
    return center_x, center_y, center_offset_x, center_offset_y, box_w, box_h


def calculate_sha256(tensor):
    if tensor.device == "cpu":
        tensor_bytes = tensor.numpy().tobytes()  # Convert tensor to a byte array
    else:
        tensor_cpu = tensor.cpu()

        tensor = tensor.detach()
        del tensor
        torch.cuda.empty_cache()

        tensor_bytes = tensor_cpu.tobytes()  # Convert tensor to a byte array

    sha256_hash = hashlib.sha256(tensor_bytes)
    return sha256_hash.hexdigest()


def get_image_features(
        image, model, preprocess, device=DEVICE,
):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        # l2 normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)

    image_features_cpu = image_features.cpu()

    # free gpu memory
    image_features = image_features.detach()
    del image_features
    torch.cuda.empty_cache()

    return image_features_cpu.numpy()


def main():
    start_time = time.time()  # Save the start time

    config = ModelPathConfig()
    sd = init_stable_diffusion(DEVICE, config, n_steps=20, sampler_name="ddim", ddim_eta=0.0)
    clip_text_embedder = CLIPTextEmbedder(device=DEVICE)
    clip_text_embedder.load_submodels()
    images = []
    prompts = []
    embedded_prompts_list = []  # To store the embedded prompts
    null_prompt_list = []  # To store the null prompts
    for i in range(NUM_ITERATIONS):
        prompt = generate_prompt()
        prompts.append(prompt)  # Store each prompt for later use
        embedded_prompt, null_prompt = embed_and_save_prompts(clip_text_embedder, prompt, i)
        torch.save(embedded_prompt, f'{EMBEDDED_PROMPTS_DIR}/embedded_prompt_{i}.pt')
        torch.cuda.empty_cache()

    images_generator = generate_images_from_disturbed_embeddings(sd, clip_text_embedder, prompts,
                                                                 batch_size=1)  # Use the corresponding prompt for each iteration

    image_encoder = CLIPImageEncoder(device=DEVICE)
    image_encoder.load_submodels()
    image_encoder.initialize_preprocessor()

    predictor = ChadScorePredictor(768, device=DEVICE)
    predictor.load_model(SCORER_CHECKPOINT_PATH)

    json_output = []
    manifest = []
    scores = []
    images_tensors = []

    json_output_path = join(FEATURES_DIR, "features.json")
    manifest_path = join(OUTPUT_DIR, "manifest.json")
    scores_path = join(OUTPUT_DIR, "scores.json")

    for i, (image, embedding, prompt_index) in enumerate(
            images_generator):  # Retrieve the prompt with the image and embedding
        # images_tensors.append(image)
        torch.cuda.empty_cache()
        get_memory_status(DEVICE)

        img_hash = calculate_sha256(image.squeeze())
        pil_image = to_pil(image.squeeze())

        prep_img = image_encoder.preprocess_input(pil_image)
        image_features = image_encoder(prep_img)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        score = predictor.model(image_features.to(DEVICE).float())

        img_file_name = f"image_{i:06d}.jpg"
        full_img_path = join(IMAGES_DIR, img_file_name)
        img_path = "./images/" + os.path.relpath(full_img_path, IMAGES_DIR)
        pil_image.save(full_img_path)

        manifest_i = {
            "file-name": img_file_name,
            "file-path": img_path,
            "file-hash": img_hash,
        }
        manifest.append(manifest_i)

        scores_i = manifest_i.copy()
        scores_i["score"] = score.item()
        scores.append(scores_i)

        numpy_img = np.array(pil_image)
        largest_bounding_box = get_bounding_box_details(numpy_img)
        center_x, center_y, center_offset_x, center_offset_y, box_w, box_h = get_bounding_box_center_offset(
            largest_bounding_box, numpy_img.shape)

        json_output_i = manifest_i.copy()
        json_output_i["prompt"] = prompts[prompt_index]  # Retrieve the prompt associated with this image
        json_output_i["score"] = score.item()
        json_output_i["cfg_strength"] = args.cfg_strength
        json_output_i["bounding_box_center_x"] = center_x
        json_output_i["bounding_box_center_y"] = center_y
        json_output_i["bounding_box_center_offset_x"] = center_offset_x
        json_output_i["bounding_box_center_offset_y"] = center_offset_y
        json_output_i["bounding_box_width"] = box_w
        json_output_i["bounding_box_height"] = box_h
        json_output_i["embedding_tensor"] = embedding.tolist()
        json_output_i["clip-vector"] = image_features.tolist()

        json_output.append(json_output_i)

    json.dump(json_output, open(json_output_path, "w"), indent=4)
    json.dump(scores, open(scores_path, "w"), indent=4)
    json.dump(manifest, open(manifest_path, "w"), indent=4)

    end_time = time.time()  # Save the end time

    print(f"Execution time: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()
