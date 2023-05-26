import os, sys
import random
import argparse

from text_to_image import Txt2Img
import torch
import time

noise_seeds = [
    2982,
    4801,
    1995,
    3598,
    987,
    3688,
    8872,
    762
]

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


# main function, called when the script is run
def generate_images(
    prompt_prefix: str="A woman with flowers in her hair in a courtyard, in the style of",
    artist_file: str='./input/prompts/artists.txt',
    output_dir: str='./outputs/noise-tests/',
    checkpoint_path: str='./input/model/sd-v1-4.ckpt',
    sampler_name: str='ddim',
    n_steps: int=20,
    num_seeds: int=8,
    noise_file: str="../input/noise-seeds.txt",
):
    time_before_initialization = time.time()
    txt2img = Txt2Img(checkpoint_path=checkpoint_path, sampler_name=sampler_name, n_steps=n_steps)

    time_after_initialization = time.time()

    # Generate images for each artist
    with open(artist_file, 'r') as f:
        artists = f.readlines()
        total_images = num_seeds * len(artists)
        for artist in artists:
            artist = artist.strip()
            prompt = generate_prompt(prompt_prefix, artist)

            save_noise_seeds(num_seeds, noise_file)

            with torch.no_grad():
                for i in range(num_seeds):
                    noise_seed = noise_seeds[i % len(noise_seeds)]

                    print("INFO: Generating image %s/%s from total" % (i + 1, total_images))
                    print("INFO: Generating image %s/%s for prompt \"%s\"" % (i+1, num_seeds, prompt))

                    # Generate image
                    image_name = f"a{i:04d}_n{noise_seed}.jpg"
                    dest_path = os.path.join(output_dir, image_name)

                    # Check if the image already exists
                    if not os.path.isfile(dest_path):
                        txt2img(dest_path=dest_path, batch_size=1, prompt=prompt, seed=noise_seed)

    # Unload the Stable Diffusion model
    del txt2img

    end_time = time.time()

    print("[SUMMARY]")
    print("Total time taken: %.2f seconds" % (end_time - time_before_initialization))
    print("Partial time (without initialization): %.2f seconds" % (end_time - time_after_initialization))
    print("Total images generated: %s" % total_images)
    print("Images/second: %.2f" % (total_images / (end_time - time_before_initialization)))
    print("Images/second (without initialization): %.2f" % (total_images / (end_time - time_after_initialization)))

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
        default='./input/artists.txt',
        help='Path to the file containing the artists, each on a line (default: \'%(default)s\')'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./outputs',
        help='Path to the output directory (default: %(default)s)'
    )

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./input/model/sd-v1-4.ckpt',
        help='Path to the checkpoint file (default: \'./input/model/sd-v1-4.ckpt\')'
    )

    parser.add_argument(
        '--sampler',
        type=str,
        default='ddim',
        help='Name of the sampler to use (default: %(default)s)'
    )

    parser.add_argument(
        '--steps',
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
        output_dir=args.output,
        checkpoint_path=args.checkpoint_path,
        sampler_name=args.sampler,
        n_steps=args.steps,
        num_seeds=args.num_seeds,
        noise_file=args.noise_file,
    )

if __name__ == "__main__":
    main()
