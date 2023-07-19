import os

from text_to_image_custom import Txt2Img
import torch
import time
from tqdm import tqdm

from stable_diffusion_reference.utils.model_custom import save_images, initialize_autoencoder, initialize_clip_embedder, initialize_unet
from stable_diffusion_reference.model.autoencoder import Encoder, Decoder, Autoencoder
from stable_diffusion_reference.model.clip_embedder import CLIPTextEmbedder
from stable_diffusion_reference.model.unet import UNetModel
from cli_builder import CLI

noise_seeds = [
    2982,
    # 4801,
    # 1995,
    # 3598,
    # 987,
    # 3688,
    # 8872,
    # 762
]
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def generate_prompt(prompt_prefix, artist):
    # Generate the prompt
    prompt = f"{prompt_prefix} {artist}"
    return prompt



def init_txt2img(checkpoint_path, sampler_name, n_steps, model_path):
    
    
    # clip_text_embedder_model = torch.load('./input/clip_embedder.pt')
    
    # print("Time to load CLIP from disk: %.2f seconds" % (t1_clip-t0_clip))
    # print("CLIP model: ", clip_text_embedder_model, type(clip_text_embedder_model))
    t0_clip = time.time()
    txt2img = Txt2Img(checkpoint_path=checkpoint_path, sampler_name=sampler_name, n_steps=n_steps)
    # txt2img.initialize_script(clip_text_embedder=clip_text_embedder_model)
    txt2img.initialize_from_saved(model_path)
    t1_clip = time.time()
    print("Time to load load the whole thing from disk: %.2f seconds" % (t1_clip-t0_clip))
    # txt2img.initialize_script()
    return txt2img

def get_all_prompts(prompt_prefix, artist_file):
    with open(artist_file, 'r') as f:
        artists = f.readlines()

    num_seeds = len(noise_seeds)
    number_of_artists = len(artists)
    total_images = num_seeds * number_of_artists

    print(f"Artist count: {number_of_artists}")
    print(f"Seed count: {num_seeds}")
    print(f"Total images: {total_images}")
    
    artists = filter(lambda a: a, map(lambda a: a.strip(), artists))
    prompts = map(lambda a: generate_prompt(prompt_prefix, a), artists)

    return total_images, prompts

def show_summary(total_time, partial_time, total_images, output_dir):
    print("[SUMMARY]")
    print("Total time taken: %.2f seconds" % total_time)
    print("Partial time (without initialization): %.2f seconds" % partial_time)
    print("Total images generated: %s" % total_images)
    print("Images/second: %.2f" % (total_images / total_time))
    print("Images/second (without initialization): %.2f" % (total_images / partial_time))

    print("Images generated successfully at", output_dir)

# main function, called when the script is run
def generate_images(
    prompt_prefix: str="A woman with flowers in her hair in a courtyard, in the style of",
    artist_file: str='./input/artists.txt',
    output_dir: str='./output/noise-tests/',
    checkpoint_path: str='./input/model/sd-v1-4.ckpt',
    model_path: str='./input/model/model.pt',
    sampler_name: str='ddim',
    n_steps: int=20,
    batch_size: int=1,
):
    time_before_initialization = time.time()
    
    txt2img = init_txt2img(checkpoint_path, sampler_name, n_steps, model_path)

    time_after_initialization = time.time()

    total_images, prompts = get_all_prompts(prompt_prefix, artist_file)

    with torch.no_grad():
        with tqdm(total=total_images, desc='Generating images', ) as pbar:
            for prompt_index, prompt in enumerate(prompts):
                for seed_index, noise_seed in enumerate(noise_seeds):
                    p_bar_description = f"Generating image {seed_index+prompt_index+1} of {total_images}"
                    pbar.set_description(p_bar_description)

                    image_name = f"n{noise_seed:04d}_a{prompt_index+1:04d}.jpg"
                    dest_path = os.path.join(output_dir, image_name)
                    
                    images = txt2img.generate_images(
                        batch_size=batch_size,
                        prompt=prompt,
                        seed=noise_seed
                    )

                    save_images(images, dest_path=dest_path)
                    
                    pbar.update(1)

    end_time = time.time()

    show_summary(
        total_time=end_time - time_before_initialization,
        partial_time=end_time - time_after_initialization,
        total_images=total_images,
        output_dir=output_dir
    )

def main():
    # args = CLI('Generate images from noise seeds.') \
    #     .prompt_prefix() \
    #     .artist_file() \
    #     .output() \
    #     .checkpoint_path() \
    #     .sampler() \
    #     .steps() \
    #     .batch_size() \
    #     .parse()

    # generate_images(
    #     prompt_prefix=args.prompt_prefix,
    #     artist_file=args.artist_file,
    #     output_dir=args.output,
    #     checkpoint_path=args.checkpoint_path,
    #     sampler_name=args.sampler,
    #     n_steps=args.steps,
    # )
    artists_file = os.path.abspath('./input/artists.txt')
    generate_images(artist_file=artists_file)
    # generate_images_from_embeddings()

if __name__ == "__main__":
    main()
