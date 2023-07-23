import os
import sys
import argparse
import torch
import hashlib
import json
import shutil
import math
import cv2
import numpy as np
import random

base_dir = "./"
sys.path.insert(0, base_dir)

from typing import List
from os.path import join

from stable_diffusion.model.clip_text_embedder import CLIPTextEmbedder
from stable_diffusion.model.clip_image_encoder import CLIPImageEncoder
from chad_score import ChadPredictor
from stable_diffusion import StableDiffusion
from stable_diffusion.constants import ModelsPathTree
from stable_diffusion.utils.utils import (
    check_device,
    get_memory_status,
    to_pil,
    save_image_grid,
    show_image_grid,
)

EMBEDDED_PROMPTS_DIR = os.path.abspath("./input/embedded_prompts/")
OUTPUT_DIR = "./output/disturbing_embeddings/"
FEATURES_DIR = join(OUTPUT_DIR, "features/")
IMAGES_DIR = join(OUTPUT_DIR, "images/")
# SCORER_CHECKPOINT_PATH = os.path.abspath("./input/model/aesthetic_scorer/sac+logos+ava1-l14-linearMSE.pth")
SCORER_CHECKPOINT_PATH = os.path.abspath("./input/model/aesthetic_scorer/chadscorer.pth")



# DEVICE = input("Set device: 'cuda:i' or 'cpu'")

prompt_list_1 = ['chibi', 'waifu', 'scifi', 'side scrolling', 'character', 'side scrolling']
prompt_list_2 = ['white background', 'centered', 'full character', 'no background', 
                 'not centered', 'line drawing', 'sketch', 'black and white', 
                 'colored', 'offset', 'video game']


parser = argparse.ArgumentParser("Embed prompts using CLIP")
parser.add_argument(
    "--prompt",
    type=str,
    help="The prompt to embed. Defaults to a randomly generated prompt",
)
parser.add_argument(
    "--save_embeddings",
    type=bool,
    default=False,
    help="If True, the disturbed embeddings will be saved to disk. Defaults to False.",
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
    default=2982,
    help="The noise seed used to generate the images. Defaults to 2982",
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
    default="cuda:0",
    help="The cuda device to use. Defaults to 'cuda:0'.",
)
parser.add_argument(
    "--clear_output_dir",
    type=bool,
    default=False,
    help="Avoid. If True, the output directory will be cleared before generating images. Defaults to False.",
)

parser.add_argument(
    "--random_walk",
    type=bool,
    default=False,
    help="Random walk on the embedding space, with the prompt embedding as origin. Defaults to False.",
)
args = parser.parse_args()

NULL_PROMPT = ""
PROMPT = args.prompt
NUM_ITERATIONS = args.num_iterations
SEED = args.seed
NOISE_MULTIPLIER = args.noise_multiplier
DEVICE = check_device(args.cuda_device)
BATCH_SIZE = args.batch_size
SAVE_EMBEDDINGS = args.save_embeddings
CLEAR_OUTPUT_DIR = args.clear_output_dir
RANDOM_WALK = args.random_walk
os.makedirs(EMBEDDED_PROMPTS_DIR, exist_ok=True)

pt = ModelsPathTree(base_directory=base_dir)


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

def init_stable_diffusion(device, pt, sampler_name="ddim", n_steps=20, ddim_eta=0.0):
    device = check_device(device)

    stable_diffusion = StableDiffusion(
        device=device, sampler_name=sampler_name, n_steps=n_steps, ddim_eta=ddim_eta
    )

    stable_diffusion.quick_initialize()
    stable_diffusion.model.load_unet(**pt.unet)
    stable_diffusion.model.load_autoencoder(**pt.autoencoder).load_decoder(**pt.decoder)

    return stable_diffusion

def generate_prompt():
    # Select 2 items randomly from prompt_list_1 and 5 items randomly from prompt_list_2
    selected_prompts_1 = random.sample(prompt_list_1, 2)
    selected_prompts_2 = random.sample(prompt_list_2, 5)

    # Join all selected prompts into a single string, separated by commas
    prompt = ', '.join(selected_prompts_1 + selected_prompts_2)
    print(f"Generated prompt: {prompt}")
    return prompt


def embed_and_save_prompts(prompt: str, i: int, null_prompt = NULL_PROMPT):

    null_prompt = null_prompt

    clip_text_embedder = CLIPTextEmbedder(device=check_device())
    clip_text_embedder.load_submodels()

    null_cond = clip_text_embedder(null_prompt)
    torch.save(null_cond, join(EMBEDDED_PROMPTS_DIR, f"null_cond_{i}.pt"))
    print(
        "Null prompt embedding saved at: ",
        f"{join(EMBEDDED_PROMPTS_DIR, f'null_cond_{i}.pt')}",
    )

    embedded_prompts = clip_text_embedder(prompt)
    torch.save(embedded_prompts, join(EMBEDDED_PROMPTS_DIR, f"embedded_prompts_{i}.pt"))
    
    print(
        "Prompts embeddings saved at: ",
        f"{join(EMBEDDED_PROMPTS_DIR, f'embedded_prompts_{i}.pt')}",
    )
    
    get_memory_status()
    clip_text_embedder.to("cpu")
    del clip_text_embedder
    torch.cuda.empty_cache()
    get_memory_status()
    return embedded_prompts, null_cond


def generate_images_from_disturbed_embeddings(
    sd: StableDiffusion,
    embedded_prompt: torch.Tensor,
    null_prompt: torch.Tensor,
    device=DEVICE,
    seed=SEED,
    num_iterations=NUM_ITERATIONS,
    noise_multiplier=NOISE_MULTIPLIER,
    batch_size=BATCH_SIZE
):
    # generator = torch.Generator(device=device).manual_seed(seed)

    # embedding_mean, embedding_std = embedded_prompt.mean(), embedded_prompt.std()
    # embedding_shape = tuple(embedded_prompt.shape)

    # noise = torch.normal(
    #     mean=embedding_mean.item(),
    #     std=embedding_std.item(),
    #     size=embedding_shape,
    #     device=device,
    #     generator=generator,
    # )
    # test with standard normal distribution
    # noise = torch.normal(
    #     mean=0.0,
    #     std=1.0,
    #     size=embedding_shape,
    #     device=device,
    #     generator=generator,
    # )
    # embedded_prompt.mean(dim=2), embedded_prompt.std(dim=2)
    # noise = torch.normal(
    #     mean=embedded_prompt.mean(dim=2), std=embedded_prompt.std(dim=2)
    # )
    dist = torch.distributions.normal.Normal(
        loc=embedded_prompt.mean(dim=2), scale=embedded_prompt.std(dim=2)
    )


    if not RANDOM_WALK:
        for i in range(0, num_iterations):

            j = num_iterations - i

            noise_i = (
                dist.sample(sample_shape=torch.Size([768])).permute(1, 0, 2).permute(0, 2, 1)
            ).to(device)
            noise_j = (
                dist.sample(sample_shape=torch.Size([768])).permute(1, 0, 2).permute(0, 2, 1)
            ).to(device)
            embedding_e = embedded_prompt + ((i * noise_multiplier) * noise_i + (j * noise_multiplier) * noise_j) / (2 * num_iterations)

            image_e = sd.generate_images_from_embeddings(
                seed=seed, 
                embedded_prompt=embedding_e, 
                null_prompt=null_prompt, 
                batch_size=batch_size
            )
            
            yield (image_e, embedding_e)
    else:
    
        noise_t = torch.zeros_like(embedded_prompt).to(device)
    
        for i in range(0, num_iterations):
            noise_i = (
                dist.sample(sample_shape=torch.Size([768])).permute(1, 0, 2).permute(0, 2, 1)
            ).to(device)
            noise_t = noise_t + noise_i
            embedding_e = embedded_prompt + (noise_multiplier * noise_t) 

            image_e = sd.generate_images_from_embeddings(
                seed=seed, 
                embedded_prompt=embedding_e, 
                null_prompt=null_prompt, 
                batch_size=batch_size
            )
            
            yield (image_e, embedding_e)

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
        tensor_bytes = tensor.cpu().numpy().tobytes()  # Convert tensor to a byte array
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
    image_features = image_features.cpu().detach().numpy()
    return image_features

def main():
    
    pt = ModelsPathTree(base_directory=base_dir)
    sd = init_stable_diffusion(DEVICE, pt, n_steps=20, sampler_name="ddim", ddim_eta=0.0)

    image_encoder = CLIPImageEncoder(device=DEVICE)
    image_encoder.load_clip_model(**pt.clip_model)
    image_encoder.initialize_preprocessor()
    
    loaded_model = torch.load(SCORER_CHECKPOINT_PATH)
    predictor = ChadPredictor(768, device=DEVICE)
    predictor.load_state_dict(loaded_model)
    predictor.eval()

    json_output = []
    manifest = []
    scores = []
    images_tensors = []
    
    json_output_path = join(FEATURES_DIR, "features.json")
    manifest_path = join(OUTPUT_DIR, "manifest.json")
    scores_path = join(OUTPUT_DIR, "scores.json")

    for i in range(NUM_ITERATIONS):
        PROMPT = generate_prompt()
        print(f"Prompt {i}: {PROMPT}")
        embedded_prompts, null_prompt = embed_and_save_prompts(PROMPT, i)
        images_generator = generate_images_from_disturbed_embeddings(sd, embedded_prompts, null_prompt, batch_size = 1)
        image, embedding = next(images_generator)
        print(f"Image {i} generated.")

        images_tensors.append(image)
        torch.cuda.empty_cache()
        get_memory_status()

        img_hash = calculate_sha256(image.squeeze())
        pil_image = to_pil(image.squeeze())

        prep_img = image_encoder.preprocess_input(pil_image)
        image_features = image_encoder(prep_img)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        score = predictor(image_features.to(DEVICE).float())
        
        img_file_name = f"image_{i:06d}.png"
        img_path = join(IMAGES_DIR, img_file_name)
        pil_image.save(img_path)

        if SAVE_EMBEDDINGS:
            embedding_file_name = f"embedding_{i:06d}.pt"
            embedding_path = join(FEATURES_DIR, embedding_file_name)
            torch.save(embedding, embedding_path)

        manifest_i = {                     
                        "file-name": img_file_name,
                        "file-hash": img_hash,
                        "file-path": img_path,
                    }
        manifest.append(manifest_i)

        scores_i = manifest_i.copy()
        scores_i["score"] = score.item()
        scores.append(scores_i)

        numpy_img = np.array(pil_image)
        largest_bounding_box = get_bounding_box_details(numpy_img)
        center_x, center_y, center_offset_x, center_offset_y, box_w, box_h = get_bounding_box_center_offset(largest_bounding_box, numpy_img.shape)

        json_output_i = manifest_i.copy()
        json_output_i["initial-prompt"] = PROMPT
        json_output_i["score"] = score.item()
        json_output_i["bounding_box_center_x"] = center_x
        json_output_i["bounding_box_center_y"] = center_y
        json_output_i["bounding_box_center_offset_x"] = center_offset_x
        json_output_i["bounding_box_center_offset_y"] = center_offset_y
        json_output_i["bounding_box_width"] = box_w
        json_output_i["bounding_box_height"] = box_h
        json_output_i["embedding-tensor"] = embedding.tolist()
        json_output_i["clip-vector"] = image_features.tolist()

        json_output.append(json_output_i)

    images_grid = torch.cat(images_tensors)
    save_image_grid(images_grid, join(IMAGES_DIR, "images_grid.png"), nrow=int(math.log(NUM_ITERATIONS, 2)), normalize=True, scale_each=True)
    
    json.dump(json_output, open(json_output_path, "w"), indent=4)
    json.dump(scores, open(scores_path, "w"), indent=4)
    json.dump(manifest, open(manifest_path, "w"), indent=4)

if __name__ == "__main__":
    main()
