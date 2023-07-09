import os, sys
import hashlib
import argparse

sys.path.append(os.path.abspath(''))
sys.path.append(os.path.abspath('./scripts'))
from scripts import text_to_image

from stable_diffusion.utils.model import save_images
from labml import monit
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import random

class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

def get_score(image, device, predictor):
    image_features = get_image_features(image, device)
    score = predictor(torch.from_numpy(image_features).to(device).float())
    return score.item()

def main():
    parser = argparse.ArgumentParser(description='Generate images from a list of prompts and score them based on aesthetic quality')
    parser.add_argument('--score_model', type=str, default='./scripts/classifiers/chadscorer.pth',
                        help='Path to the aesthetic score model file')
    parser.add_argument('--prompts_file', type=str, default='./input/prompts.txt',
                        help='Path to the prompts file')
    parser.add_argument('--seeds_file', type=str, default=None,
                        help='Path to the seeds file')
    parser.add_argument('--output', type=str, default='./output/chadscore',
                        help='Output directory for saving the scored images')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force the script to use CPU instead of GPU')
    parser.add_argument('--checkpoint_path', type=str, default="./input/model/sd-v1-4.ckpt",
                        help='Path to stable diffusion checkpoint file to use for image generation')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='How many images to generate per prompt/seed combo')
    parser.add_argument('--steps', type=int, default=20,
                        help='Steps to diffuse the image when generating')
    parser.add_argument('--scale', type=int, default=512,
                        help='Resolution for the generated images')
    args = parser.parse_args()

    # Load the aesthetic predictor model

    args.score_model = os.path.abspath(args.score_model)
    args.prompts_file = os.path.abspath(args.prompts_file)
    if args.seeds_file:
        args.seeds_file = os.path.abspath(args.seeds_file)
    args.output = os.path.abspath(args.output)
    args.checkpoint_path = os.path.abspath(args.checkpoint_path)
    device = "cpu" if args.force_cpu else "cuda" if torch.cuda.is_available() else "cpu"
    pt_state = torch.load(args.score_model, map_location=torch.device('cpu'))
    predictor = AestheticPredictor(768)
    predictor.load_state_dict(pt_state)
    predictor.to(device)
    predictor.eval()

    # Create the output directory if it doesn't exist
    Path(args.output).mkdir(exist_ok=True)

    # Read the prompts and seeds from the files
    with open(args.prompts_file, "r") as filec:
        prompts = filec.read().split("\n")
    seeds = [random.randint(1, 9999) for _ in range(len(prompts))] if args.seeds_file is None else []
    print("INFO: This script will generate a total of %s images" % (args.batch_size * len(prompts)))

    text2img = text_to_image.Txt2Img(checkpoint_path=args.checkpoint_path,
                                        n_steps=args.steps,
                                     )
    text2img.initialize_script()

    # Read the seeds from the file if provided
    if args.seeds_file is not None:
        with open(args.seeds_file, "r") as files:
            seeds = files.read().split("\n")

    # Generate and score the images
    with monit.section('Generate', total_steps=len(prompts)) as section:
        for i in tqdm(range(len(prompts))):
            prompt = prompts[i]
            seed = seeds[i % len(seeds)]
            print("\nINFO: Currently generating %s images for prompt \"%s\"" % (args.batch_size, prompt))
            print("Progress: %s/%s (%.2f%%) (from total prompts)" % (i, len(prompts), (i/len(prompts))*100))

            # Generate image
            prompt = prompt.strip()

            image = text2img.generate_images(
                        prompt=prompt,
                        batch_size=args.batch_size,
                        h=args.scale,
                        w=args.scale,
                    )

            # Get the aesthetic score
            score = get_score(image, device, predictor)

            # Save the image in the appropriate bin folder based on the score
            bin_folder = Path(args.output) / f"{int(score)}"
            bin_folder.mkdir(exist_ok=True)
            image_path = bin_folder / f"image_{i}.png"
            save_images(image, image_path)
            section.progress(1)

    print("Images scored and saved successfully.")

if __name__ == '__main__':
    main()
