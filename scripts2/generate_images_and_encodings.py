import os
import sys
import shutil
import torch
import time
from tqdm import tqdm
import numpy as np

base_directory = "./"
sys.path.insert(0, base_directory)

from stable_diffusion2.stable_diffusion import StableDiffusion
from stable_diffusion2.model.clip_image_encoder import CLIPImageEncoder
from stable_diffusion2.utils.utils import save_images, check_device

from cli_builder import CLI
from PIL import Image

noise_seeds = [2982, 4801, 1995, 3598, 987, 3688, 8872, 762]


def init_stable_diffusion(device, sampler_name="ddim", n_steps="20", ddim_eta=0.0):
    device = check_device(device)
    stable_diffusion = StableDiffusion(
        device=device, sampler_name=sampler_name, n_steps=n_steps, ddim_eta=ddim_eta
    )

    stable_diffusion.quick_initialize().load_submodel_tree()
    # stable_diffusion.model.load_clip_embedder()
    # stable_diffusion.model.load_unet()
    return stable_diffusion


def show_summary(total_time, partial_time, total_images, output_dir):
    print("[SUMMARY]")
    print("Total time taken: %.2f seconds" % total_time)
    print("Partial time (without initialization): %.2f seconds" % partial_time)
    print("Total images generated: %s" % total_images)
    print("Images/second: %.2f" % (total_images / total_time))
    print(
        "Images/second (without initialization): %.2f" % (total_images / partial_time)
    )

    print("Images generated successfully at", output_dir)


# main function, called when the script is run
def generate_images(
    prompt: str = "An oil painting of a computer generated image of a geometric pattern",
    output_base_dir: str = "./output/stable_diffusion/",
    sampler_name: str = "ddim",
    n_steps: int = 20,
    batch_size: int = 1,
    num_iterations: int = 10,
    device=None,
):
    images_dir = os.path.join(output_base_dir, "images/")
    shutil.rmtree(images_dir, ignore_errors=True)
    os.makedirs(images_dir, exist_ok=True)

    clip_vectors_dir = os.path.join(output_base_dir, "features/")
    shutil.rmtree(clip_vectors_dir, ignore_errors=True)
    os.makedirs(clip_vectors_dir, exist_ok=True)

    stable_diffusion = init_stable_diffusion(device, sampler_name, n_steps)
    image_encoder = CLIPImageEncoder(device=device)
    # image_encoder.load_submodels()
    image_encoder.load_clip_model()
    image_encoder.initialize_preprocessor()

    with torch.no_grad():
        image_counter = 0
        for i in range(num_iterations):
            images = stable_diffusion.generate_images(
                batch_size=batch_size,
                prompt=prompt,
            )
            # print(f"{images.mean(), images.std()}")
            images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
            # Transpose to `[batch_size, height, width, channels]` and convert to numpy
            images = images.cpu()
            images = images.permute(0, 2, 3, 1)
            images = images.float().numpy()
            # Save images
            for j, img in enumerate(images):
                image_name = f"{image_counter:06d}.jpg"
                img_dest_path = os.path.join(images_dir, image_name)
                img = Image.fromarray((255.0 * img).astype(np.uint8))
                
                img.save(img_dest_path)

                prep_img = image_encoder.preprocess_input(img)
                clip_vector = image_encoder(prep_img)
                torch.save(
                    clip_vector,
                    os.path.join(clip_vectors_dir, f"{image_counter:06d}.pt"),
                )

                image_counter += 1


def main():
    args = (
        CLI("Generate images from noise seeds.")
        .prompt()
        .batch_size()
        .num_iterations()
        .cuda_device()
        .parse()
    )

    generate_images(
        prompt=args.prompt,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        device=args.cuda_device,
    )


if __name__ == "__main__":
    main()
