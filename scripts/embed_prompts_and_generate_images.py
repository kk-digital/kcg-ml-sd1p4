import os
import sys
import argparse
import torch
import hashlib
import json
import clip
import shutil
import math

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
OUTPUT_DIR = os.path.abspath("./output/disturbing_embeddings/")
FEATURES_DIR = os.path.abspath(join(OUTPUT_DIR, "features/"))
IMAGES_DIR = os.path.abspath(join(OUTPUT_DIR, "images/"))
# SCORER_CHECKPOINT_PATH = os.path.abspath("./input/model/aesthetic_scorer/sac+logos+ava1-l14-linearMSE.pth")
SCORER_CHECKPOINT_PATH = os.path.abspath("./input/model/aesthetic_scorer/chadscorer.pth")



# DEVICE = input("Set device: 'cuda:i' or 'cpu'")

pt = ModelsPathTree(base_directory=base_dir)



parser = argparse.ArgumentParser("Embed prompts using CLIP")
parser.add_argument(
    "--prompt",
    type=str,
    default='A woman with flowers in her hair in a courtyard, in the style of Frank Frazetta',
    help="The prompt to embed. Defaults to 'A woman with flowers in her hair in a courtyard, in the style of Frank Frazetta'",
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



def embed_and_save_prompts(prompt: str, null_prompt = NULL_PROMPT):

    null_prompt = null_prompt
    prompt = prompt

    clip_text_embedder = CLIPTextEmbedder(device=check_device())
    clip_text_embedder.load_submodels()

    null_cond = clip_text_embedder(null_prompt)
    torch.save(null_cond, join(EMBEDDED_PROMPTS_DIR, "null_cond.pt"))
    print(
        "Null prompt embedding saved at: ",
        f"{join(EMBEDDED_PROMPTS_DIR, 'null_cond.pt')}",
    )

    embedded_prompts = clip_text_embedder(prompt)
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


    for i in range(0, num_iterations):

        j = num_iterations - i

        noise_i = (
            dist.sample(sample_shape=torch.Size([768])).permute(1, 0, 2).permute(0, 2, 1)
        )
        noise_j = (
            dist.sample(sample_shape=torch.Size([768])).permute(1, 0, 2).permute(0, 2, 1)
        )
        embedding_e = embedded_prompt + ((i * noise_multiplier) * noise_i + (j * noise_multiplier) * noise_j) / (2 * num_iterations)
        
        image_e = sd.generate_images_from_embeddings(
            seed=seed, embedded_prompt=embedding_e, null_prompt=null_prompt, batch_size=batch_size
        )
        
        yield (image_e, embedding_e)


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

    embedded_prompts, null_prompt = embed_and_save_prompts(PROMPT)
    
    sd = StableDiffusion(device=DEVICE)
    sd.quick_initialize().load_autoencoder(**pt.autoencoder).load_decoder(**pt.decoder)
    sd.model.load_unet(**pt.unet)

    images = generate_images_from_disturbed_embeddings(sd, embedded_prompts, null_prompt, batch_size = 1)
    
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=DEVICE)
    
    loaded_model = torch.load(SCORER_CHECKPOINT_PATH)
    predictor = ChadPredictor(768, device=DEVICE)
    predictor.load_state_dict(loaded_model)
    predictor.eval()

    json_output = []
    manifest = []

    images_tensors = []
    for i, (image, embedding) in enumerate(images):
        images_tensors.append(image)
        #compute hash
        img_hash = calculate_sha256(image.squeeze())
        pil_image = to_pil(image.squeeze())
        #compute aesthetic score
        image_features = get_image_features(pil_image, clip_model, clip_preprocess)
        score = predictor(torch.from_numpy(image_features).to(DEVICE).float())
        img_file_name = f"image_{i}.png"
        img_path = join(IMAGES_DIR, img_file_name)
        pil_image.save(img_path)
        print(f"Image saved at: {img_path}")

        if SAVE_EMBEDDINGS:
            embedding_file_name = f"embedding_{i}.pt"
            embedding_path = join(FEATURES_DIR, embedding_file_name)
            torch.save(embedding, embedding_path)
            print(f"Embedding saved at: {embedding_path}")

        manifest_i =    {                     
                            "file-name": img_file_name,
                            "file-hash": img_hash,
                            "file-path": img_path,
                            "aesthetic-score": score.item(),
                        }
        manifest.append(manifest_i)

        json_output_i = manifest_i.copy()
        json_output_i["initial-prompt"] = PROMPT
        json_output_i["embedding-tensor"] = embedding.tolist()
        json_output_i["clip-vector"] = image_features.tolist()
        json_output.append(json_output_i)

        # json_output.append( 
        #                     {
        #                         "file-name": img_file_name,
        #                         "file-hash": img_hash,
        #                         "file-path": img_path,
        #                         "aesthetic-score": score.item(),
        #                         "initial-prompt": PROMPT,
        #                         "embedding-tensor": embedding.tolist(),
        #                         "clip-vector": image_features.tolist()
        #                     }
        #                 )
        
        # manifest.append( 
        #                 {
        #                     "file-name": img_file_name,
        #                     "file-hash": img_hash,
        #                     "file-path": img_path,
        #                     "aesthetic-score": score.item(),
        #                 }
        #             )

    images_grid = torch.cat(images_tensors)
    save_image_grid(images_grid, join(IMAGES_DIR, "images_grid.png"), nrow=int(math.log(NUM_ITERATIONS, 2)), normalize=True, scale_each=True)
    print(f"Image grid saved at: {join(IMAGES_DIR, 'images_grid.png')}")
    json_output_path = join(FEATURES_DIR, "features.json")
    manifest_path = join(OUTPUT_DIR, "manifest.json")
    json.dump(json_output, open(json_output_path, "w"), indent=4)
    print(f"features.json saved at: {json_output_path}")
    json.dump(manifest, open(manifest_path, "w"), indent=4)
    print(f"manifest.json saved at: {manifest_path}")


if __name__ == "__main__":
    main()




