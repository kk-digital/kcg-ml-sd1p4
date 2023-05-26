import os, sys
import random
import argparse

from text_to_image import Txt2Img
import torch


# Function to generate a prompt
def generate_prompt(prompt_prefix, artist):
    # Generate the prompt
    prompt = f"{prompt_prefix} {artist}"
    return prompt

# Function to save noise seeds to a file
def save_noise_seeds(num_seeds, output_file):
    noise_seeds = [random.randint(0, 9999) for _ in range(num_seeds)]
    with open(output_file, 'w') as f:
        for seed in noise_seeds:
            f.write(f"{seed}\n")

# Function to convert relative path to absolute path
def to_absolute_path(path: str):
    return os.path.join(os.path.dirname(__file__), path)

# main function, called when the script is run
def generate_images(
    prompt_prefix: str="A woman with flowers in her hair in a courtyard, in the style of",
    artist_file: str=to_absolute_path('../input/artists.txt'),
    output_dir: str=to_absolute_path('../output/noise-tests/'),
    checkpoint_path: str=to_absolute_path('../input/model/sd-v1-4.ckpt'),
    sampler_name: str='ddim',
    n_steps: int=20,
    num_seeds: int=8,
    noise_file: str="../input/noise-seeds.txt",
):
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("WARNING: You are running this script without CUDA. Brace yourself for a slow ride.")

    txt2img = Txt2Img(checkpoint_path=checkpoint_path, sampler_name=sampler_name, n_steps=n_steps)

    # Generate images for each artist
    with open(artist_file, 'r') as f:
        artists = f.readlines()
        for artist in artists:
            artist = artist.strip()
            prompt = generate_prompt(prompt_prefix, artist)

            save_noise_seeds(num_seeds, noise_file)

            with torch.no_grad():
                for i in range(num_seeds):
                    # Use seed
                    with open(noise_file, 'r') as f:
                        noise_seed = int(f.readlines()[i].strip())

                    # Generate image
                    image_name = f"a{i:04d}_n{noise_seed}.jpg"
                    dest_path = os.path.join(output_dir, image_name)

                    # Check if the image already exists
                    if not os.path.isfile(dest_path):
                        txt2img(dest_path=dest_path, batch_size=1, prompt=prompt, seed=noise_seed)

    # Unload the Stable Diffusion model
    del txt2img

    print("Images generated successfully at", output_dir)

def main():
    parser = argparse.ArgumentParser(description='Generate images from noise seeds.')

    parser.add_argument(
        '--prompt_prefix',
        type=str,
        default="A woman with flowers in her hair in a courtyard, in the style of",
        help='Prefix for the prompt, must end with "in the style of" (default: %(default)s)'
    )

    parser.add_argument(
        '--artist_file',
        type=str,
        default=to_absolute_path('../input/prompts/artists.txt'),
        help='Path to the file containing the artists, each on a line (default: \'../input/prompts/artists.txt\')'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=to_absolute_path('../output/noise-tests/'),
        help='Path to the output directory (default: \'../output/noise-tests/\')'
    )

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=to_absolute_path('../input/model/sd-v1-4.ckpt'),
        help='Path to the checkpoint file (default: \'../input/model/sd-v1-4.ckpt\')'
    )

    parser.add_argument(
        '--sampler_name',
        type=str,
        default='ddim',
        help='Name of the sampler to use (default: %(default)s)'
    )

    parser.add_argument(
        '--n_steps',
        type=int,
        default=20,
        help='Number of steps to use (default: %(default)s)'
    )

    parser.add_argument(
        '--num_seeds',
        type=int,
        default=8,
        help='Number of seeds to use (default: %(default)s)'
    )

    parser.add_argument(
        '--noise_file',
        type=str,
        default="noise-seeds.txt",
        help='Path to the file containing the noise seeds, each on a line (default: \'noise-seeds.txt\')'
    )

    args = parser.parse_args()

    generate_images(
        prompt_prefix=args.prompt_prefix,
        artist_file=args.artist_file,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        sampler_name=args.sampler_name,
        n_steps=args.n_steps,
        num_seeds=args.num_seeds,
        noise_file=args.noise_file,
    )

if __name__ == "__main__":
    main()
