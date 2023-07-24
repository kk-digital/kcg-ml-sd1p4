import os
import sys
import argparse
import torch
import hashlib
import json
import clip
import shutil
import math
import random
import numpy.random as npr

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
    get_device,
    get_memory_status,
    to_pil,
    save_image_grid,
    show_image_grid,
    set_seed
)

EMBEDDED_PROMPTS_DIR = os.path.abspath("./input/embedded_prompts/")
OUTPUT_DIR = "./output/data/"
FEATURES_DIR = join(OUTPUT_DIR, "features/")
IMAGES_DIR = join(OUTPUT_DIR, "images/")
# SCORER_CHECKPOINT_PATH = os.path.abspath("./input/model/aesthetic_scorer/sac+logos+ava1-l14-linearMSE.pth")
SCORER_CHECKPOINT_PATH = os.path.abspath("./input/model/aesthetic_scorer/chadscorer.pth")



# DEVICE = input("Set device: 'cuda:i' or 'cpu'")





parser = argparse.ArgumentParser("Embed prompts using CLIP")

parser.add_argument(
    "--prompts",
    nargs="+",
    type=str,
    default=['A woman with flowers in her hair in a courtyard, in the style of Frank Frazetta', 'An oil painting of a computer virus', 'chibi, waifu'],
    help="The prompts to embed. Defaults to ['A woman with flowers in her hair in a courtyard, in the style of Frank Frazetta', 'A computer virus painting a cow']",
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
    "--max_noise_steps",
    type=int,
    default=5,
    help="The maximum length of the random walk over the initial prompt. Defaults to 5.",
)
parser.add_argument(
    "--ddim_steps",
    type=int,
    default=20,
    help="Number of denoising steps during the sampling process. Defaults to 20.",
)
# parser.add_argument(
#     "--batch_size",
#     type=str,
#     default=1,
#     help="The number of images to generate per batch. Defaults to 1.",
# )

parser.add_argument(
    "--seed",
    type=int,
    default=2982,
    help="The noise seed used to generate the images. Defaults to `2982`. Set to `0` for a random seed.",
)
parser.add_argument(
    "--noise_multiplier",
    type=float,
    default=0.01,
    help="The multiplier for the amount of noise used to disturb the prompt embedding. Defaults to 0.01.",
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
PROMPTS = args.prompts
NUM_ITERATIONS = args.num_iterations
SEED = args.seed
NOISE_MULTIPLIER = args.noise_multiplier
DEVICE = get_device(args.cuda_device)
# BATCH_SIZE = args.batch_size
BATCH_SIZE = 1
SAVE_EMBEDDINGS = args.save_embeddings
CLEAR_OUTPUT_DIR = args.clear_output_dir
RANDOM_WALK = args.random_walk
MAX_NOISE_STEPS = args.max_noise_steps
DDIM_STEPS = args.ddim_steps
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

def init_stable_diffusion(device, pt, sampler_name="ddim", n_steps=DDIM_STEPS, ddim_eta=0.0):
    device = get_device(device)

    stable_diffusion = StableDiffusion(
        device=device, sampler_name=sampler_name, n_steps=n_steps, ddim_eta=ddim_eta
    )

    stable_diffusion.quick_initialize()
    stable_diffusion.model.load_unet(**pt.unet)
    stable_diffusion.model.load_autoencoder(**pt.autoencoder).load_decoder(**pt.decoder)

    return stable_diffusion

def embed_and_save_prompts(prompts: str, null_prompt = NULL_PROMPT):

    null_prompt = null_prompt
    prompts = prompts

    clip_text_embedder = CLIPTextEmbedder(device=get_device(DEVICE))
    clip_text_embedder.load_submodels()

    null_cond = clip_text_embedder(null_prompt)
    torch.save(null_cond, join(EMBEDDED_PROMPTS_DIR, "null_cond.pt"))
    print(
        "Null prompt embedding saved at: ",
        f"{join(EMBEDDED_PROMPTS_DIR, 'null_cond.pt')}",
    )

    embedded_prompts = clip_text_embedder(prompts)
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
    return embedded_prompts, null_cond

def generate_image_from_disturbed_embeddings(
    sd: StableDiffusion,
    embedded_prompts: torch.Tensor,
    null_prompt: torch.Tensor,
    device=DEVICE,
    seed=SEED,
    num_iterations=NUM_ITERATIONS,
    noise_multiplier=NOISE_MULTIPLIER,
    max_noise_steps = MAX_NOISE_STEPS,
    batch_size=BATCH_SIZE
):
    
    num_prompts = len(embedded_prompts)
    prompt_index = random.choice(range(0, num_prompts))
    num_noise_steps = random.choice(range(0, max_noise_steps))

    print("prompt index: ", prompt_index)
    print("num noise steps: ", num_noise_steps)
    embedded_prompt = embedded_prompts[prompt_index].to(device).unsqueeze(0)
   
    dist = torch.distributions.normal.Normal(
        loc=embedded_prompt.mean(dim=2), scale=embedded_prompt.std(dim=2)
    )
    noise_t = torch.zeros_like(embedded_prompt).to(device)
    for i in range(num_noise_steps):
        noise_i = (
            dist.sample(sample_shape=torch.Size([768]))
        ).permute(1, 0, 2).permute(0, 2, 1).to(device)
        print(noise_i.shape)
        noise_t = noise_t + (noise_multiplier * noise_i)   

    embedding_e = embedded_prompt + noise_t
    image_e = sd.generate_images_from_embeddings(
        seed=seed, 
        embedded_prompt=embedding_e, 
        null_prompt=null_prompt, 
        batch_size=batch_size
    )
        
    return (image_e, embedding_e, prompt_index, num_noise_steps)

def calculate_sha256(tensor):
    if tensor.device == "cpu":
        tensor_bytes = tensor.numpy().tobytes()  # Convert tensor to a byte array
    else:
        tensor_bytes = tensor.cpu().numpy().tobytes()  # Convert tensor to a byte array
    sha256_hash = hashlib.sha256(tensor_bytes)
    return sha256_hash.hexdigest()


if __name__ == "__main__":

    pt = ModelsPathTree(base_directory=base_dir)

    embedded_prompts, null_prompt = embed_and_save_prompts(PROMPTS)
    num_prompts = len(embedded_prompts)
    sd = init_stable_diffusion(DEVICE, pt, n_steps=DDIM_STEPS)    
    
    image_encoder = CLIPImageEncoder(device=DEVICE)
    image_encoder.load_clip_model(**pt.clip_model)
    image_encoder.initialize_preprocessor()
    
    loaded_model = torch.load(SCORER_CHECKPOINT_PATH)
    predictor = ChadPredictor(768, device=DEVICE)
    predictor.load_state_dict(loaded_model)
    predictor.eval()

    json_output = []
    scores = []
    manifest = []
    json_output_path = join(FEATURES_DIR, "features.json")
    manifest_path = join(OUTPUT_DIR, "manifest.json")
    scores_path = join(OUTPUT_DIR, "scores.json")
    # images_tensors = []
    # prompt_index = npr.choice(num_prompts)
    # num_noise_steps = npr.choice(MAX_NOISE_STEPS)
    img_counter = 0
    for i in range(0, NUM_ITERATIONS):

        prompt_index = random.choice(range(0, num_prompts))
        num_noise_steps = random.choice(range(0, MAX_NOISE_STEPS))

        print("prompt index: ", prompt_index)
        print("num noise steps: ", num_noise_steps)

        embedded_prompt = embedded_prompts[prompt_index].to(DEVICE).unsqueeze(0)

        dist = torch.distributions.normal.Normal(
            loc=embedded_prompt.mean(dim=2), scale=embedded_prompt.std(dim=2)
        )
        
        noise_t = torch.zeros_like(embedded_prompt).to(DEVICE)

        for k in range(num_noise_steps):
            noise_i = (
                dist.sample(sample_shape=torch.Size([768]))
            ).permute(1, 0, 2).permute(0, 2, 1).to(DEVICE)
            noise_t += (NOISE_MULTIPLIER * noise_i)

        embedding = embedded_prompt + noise_t
        image = sd.generate_images_from_embeddings(
            seed=SEED, 
            embedded_prompt=embedding, 
            null_prompt=null_prompt, 
            batch_size=BATCH_SIZE
        )        
        # images_tensors.append(image.cpu().detach())
        get_memory_status()
        #compute hash
        img_hash = calculate_sha256(image.squeeze())
        pil_image = to_pil(image.squeeze())
        #compute aesthetic score
        prep_img = image_encoder.preprocess_input(pil_image)
        image_features = image_encoder(prep_img)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        score = predictor(image_features.to(DEVICE).float()).cpu()
        img_file_name = f"image_{img_counter:06d}.png"
        img_path = join(IMAGES_DIR, img_file_name)
        pil_image.save(img_path)
        print(f"Image saved at: {img_path}")

        if SAVE_EMBEDDINGS:
            embedding_file_name = f"embedding_{img_counter:06d}.pt"
            embedding_path = join(FEATURES_DIR, embedding_file_name)
            torch.save(embedding, embedding_path)
            print(f"Embedding saved at: {embedding_path}")

        manifest_i = {                     
                        "file-name": img_file_name,
                        "file-path": "./images/"+img_file_name,
                        "file-hash": img_hash,
                    }
        manifest.append(manifest_i)

        scores_i = manifest_i.copy()
        scores_i["initial-prompt"] = PROMPTS[prompt_index]
        scores_i["initial-prompt-index"] = prompt_index
        scores_i["score"] = score.item()
        scores_i["num-noise-steps"] = num_noise_steps
        scores.append(scores_i)

        json_output_i = scores_i.copy()
        json_output_i["embedding-tensor"] = embedding.tolist()
        json_output_i["clip-vector"] = image_features.tolist()
        json_output.append(json_output_i)

        if i%64 == 0:
            json.dump(json_output, open(json_output_path, "w"), indent=4)
            print(f"features.json saved at: {json_output_path}")
            
            json.dump(scores, open(scores_path, "w"), indent=4)
            print(f"scores.json saved at: {scores_path}")
            
            json.dump(manifest, open(manifest_path, "w"), indent=4)
            print(f"manifest.json saved at: {manifest_path}")
        img_counter += 1
    json.dump(json_output, open(json_output_path, "w"), indent=4)
    print(f"features.json saved at: {json_output_path}")
    json.dump(scores, open(scores_path, "w"), indent=4)
    print(f"scores.json saved at: {scores_path}")
    json.dump(manifest, open(manifest_path, "w"), indent=4)
    print(f"manifest.json saved at: {manifest_path}")