import json
import os
import shutil
import sys
from os.path import join
import numpy as np
import torch
from PIL import Image

base_directory = "./"
sys.path.insert(0, base_directory)

from stable_diffusion.stable_diffusion import StableDiffusion
from stable_diffusion.model.clip_image_encoder import CLIPImageEncoder
from utility.utils_argument_parsing import get_seed_array_from_string
from cli_builder import CLI
from stable_diffusion.utils_backend import get_device
from stable_diffusion.utils_image import calculate_sha256

OUTPUT_DIR = "/output/stable_diffusion/"
FEATURES_DIR = join(OUTPUT_DIR, "features/")
IMAGES_DIR = join(OUTPUT_DIR, "images/")


def init_stable_diffusion(device, sampler_name="ddim", n_steps=20, ddim_eta=0.0):
    device = get_device(device)
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
        negative_prompt: str = '',
        output_base_dir: str = OUTPUT_DIR,
        sampler_name: str = "ddim",
        n_steps: int = 20,
        batch_size: int = 1,
        num_iterations: int = 10,
        seed_array = [0],
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
    image_encoder.load_submodels()
    image_encoder.initialize_preprocessor()
    manifest = []
    features = []
    with torch.no_grad():
        image_counter = 0
        for i in range(num_iterations):
            tensor_images = stable_diffusion.generate_images(
                batch_size=batch_size,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed_array[image_counter % len(seed_array)]
            )
            # print(tensor_images.shape)
            # image_hash = calculate_sha256(images.squeeze())
            # print(f"{images.mean(), images.std()}")
            images = torch.clamp((tensor_images + 1.0) / 2.0, min=0.0, max=1.0)
            # Transpose to `[batch_size, height, width, channels]` and convert to numpy
            images = images.cpu()
            images = images.permute(0, 2, 3, 1)
            images = images.float().numpy()
            # Save images
            for j, img in enumerate(images):
                print(tensor_images[j].squeeze().shape)
                image_hash = calculate_sha256(tensor_images[j].squeeze())
                image_name = f"{image_counter:06d}.jpg"
                img_dest_path = os.path.join(images_dir, image_name)
                # img_hash = hashlib.sha256(img.tobytes())
                img = Image.fromarray((255.0 * img).astype(np.uint8))

                img.save(img_dest_path)
                print(f"Saved image at {img_dest_path}.jpg")
                prep_img = image_encoder.preprocess_input(img)
                clip_vector = image_encoder(prep_img)
                # clip_vector_dest_path = os.path.abspath(os.path.join(clip_vectors_dir, f"{image_counter:06d}.pt"))
                # torch.save(
                #     clip_vector,
                #     clip_vector_dest_path,
                # )
                # print(f"Saved clip vector at {clip_vector_dest_path}")
                manifest_img_path = "./images/" + image_name
                manifest_i = {
                    "file-name": image_name,
                    "file-hash": image_hash,
                    "file-path": manifest_img_path,
                    # "clip-vector-path": clip_vector_dest_path,
                }
                features_i = manifest_i.copy()
                features_i["clip-vector"] = clip_vector.tolist()

                features.append(features_i)
                manifest.append(manifest_i)
                image_counter += 1
                if image_counter % 64 == 0:
                    manifest_path = os.path.join(output_base_dir, "manifest.json")
                    features_path = os.path.join(FEATURES_DIR, "features.json")
                    json.dump(manifest, open(manifest_path, "w"), indent=4)
                    json.dump(features, open(features_path, "w"), indent=4)

    manifest_path = os.path.join(output_base_dir, "manifest.json")
    features_path = os.path.join(FEATURES_DIR, "features.json")
    json.dump(manifest, open(manifest_path, "w"), indent=4)
    json.dump(features, open(features_path, "w"), indent=4)


def main():
    args = (
        CLI("Generate images and encodings")
        .prompt()
        .negative_prompt()
        .batch_size()
        .num_iterations()
        .cuda_device()
        .low_vram()
        .sampler()
        .cfg_scale()
        .seed()
        .parse()
    )

    # parse the seed string
    seed_array = get_seed_array_from_string(args.seed, array_size=(args.num_iterations))

    # if low vram flag is set, make sure the batch size is allways 1
    if (args.low_vram):
        args.batch_size = 1

    generate_images(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        seed_array=seed_array,
        sampler_name=args.sampler,
        device=args.cuda_device,
    )


if __name__ == "__main__":
    main()
