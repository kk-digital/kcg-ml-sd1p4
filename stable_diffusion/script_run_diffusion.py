import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

from diffusion import model, data, generate, utils


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_images', type=int, default=10, help='Number of images to generate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    args = parser.parse_args()

    # Load the diffusion model
    model_path = '/input/models/v1-5-pruned-emaonly.safetensors'
    diffusion = model.load(model_path)

    # Load the prompts
    prompts_path = '/input/prompts.txt'
    with open(prompts_path, 'r') as f:
        prompts = f.readlines()

    # Generate the images
    device = torch.device('cuda')
    batch_size = args.batch_size
    for i in tqdm(range(0, args.num_images, batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        images = generate_images(diffusion, device, batch_prompts)
        for j, img in enumerate(images):
            img_path = os.path.join('/output', f'image_{i+j:04d}.png')
            utils.save_image(img, img_path)


def generate_images(diffusion, device, prompts):
    # Convert the prompts to tensors
    tokens_list = [data.tokenize(prompt.strip()) for prompt in prompts]
    prompts_t = data.pack_sequence([torch.tensor(tokens) for tokens in tokens_list], device)

    # Generate the images
    with torch.no_grad():
        images = generate.generate_images_inpainting(
            diffusion, device, prompts_t, image_size=256, clip_min=-1., clip_max=1.)
    return images.cpu().numpy()


if __name__ == '__main__':
    main()
