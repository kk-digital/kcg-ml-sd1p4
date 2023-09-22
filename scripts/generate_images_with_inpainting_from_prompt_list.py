from datetime import datetime
import sys
from tqdm import tqdm
import os
from PIL import Image

import argparse
base_directory = "./"
sys.path.insert(0, base_directory)

from cli_builder import CLI
from utility.dataset.prompt_list_dataset import PromptListDataset
from utility.utils_logger import logger
from scripts.inpaint_A1111 import img2img


def parse_args():
    parser = argparse.ArgumentParser(description="Call img2img with specified parameters.")

    # Required parameters
    parser.add_argument("--prompt_list_dataset_path", type=str, help="The path of the prompt list dataset")
    parser.add_argument("--num_images", type=int, default=100, help="The number of images to generate")

    parser.add_argument("--init_img", type=str, help="Path to the initial image")
    parser.add_argument("--init_mask", type=str, help="Path to the initial mask")

    # Optional parameters with default values
    parser.add_argument("--sampler_name", type=str, default="ddim", help="Sampler name")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--n_iter", type=int, default=1, help="Number of iterations")
    parser.add_argument("--steps", type=int, default=20, help="Steps")
    parser.add_argument("--cfg_scale", type=float, default=7.0, help="Config scale")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--mask_blur", type=int, default=4, help="Mask blur value")
    parser.add_argument("--inpainting_fill", type=int, default=1, help="Inpainting fill value")

    # Additional parameters for StableDiffusionProcessingImg2Img
    # Add default values as needed
    parser.add_argument("--outpath", type=str, default=f"output/inpainting/{datetime.now().strftime('%m-%d-%Y')}",
                        help="Output path for samples")
    parser.add_argument("--styles", nargs="*", default=[], help="Styles list")
    parser.add_argument("--resize_mode", type=int, default=0, help="Resize mode")
    parser.add_argument("--denoising_strength", type=float, default=0.75, help="Denoising strength")
    parser.add_argument("--image_cfg_scale", type=float, default=1.5, help="Image config scale")
    parser.add_argument("--inpaint_full_res_padding", type=int, default=32, help="Inpaint full resolution padding")
    parser.add_argument("--inpainting_mask_invert", type=int, default=0, help="Inpainting mask invert value")

    return parser.parse_args()

def main():
    args = parse_args()

    num_images = args.num_images

    # load prompt list
    limit = num_images
    prompt_dataset = PromptListDataset()
    prompt_dataset.load_prompt_list(args.prompt_list_dataset_path, limit)

    # raise error when prompt list is not enough
    if len(prompt_dataset.prompt_paths) != num_images:
        raise Exception("Number of prompts do not match number of image to generate")

    init_image = Image.open(args.init_img)
    init_mask = Image.open(args.init_mask)

    # Displaying the parameters using the logger
    logger.info("Parameters for img2img:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")

    # Create the output directory if it does not exist
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)

    for i in tqdm(range(len(prompt_dataset.prompt_paths))):
        prompt_data = prompt_dataset.get_prompt_data(i)
        positive_prompt = prompt_data.positive_prompt_str
        negative_prompt = prompt_data.negative_prompt_str
        img2img(prompt=positive_prompt,
                negative_prompt=negative_prompt,
                sampler_name=args.sampler_name,
                batch_size=args.batch_size,
                n_iter=args.n_iter,
                steps=args.steps,
                cfg_scale=args.cfg_scale,
                width=args.width,
                height=args.height,
                mask_blur=args.mask_blur,
                inpainting_fill=args.inpainting_fill,
                outpath=args.outpath,
                styles=args.styles,
                init_images=[init_image],
                mask=init_mask,
                resize_mode=args.resize_mode,
                denoising_strength=args.denoising_strength,
                image_cfg_scale=args.image_cfg_scale,
                inpaint_full_res_padding=args.inpaint_full_res_padding,
                inpainting_mask_invert=args.inpainting_mask_invert
                )

if __name__ == "__main__":
    main()
