import os
import argparse
import hashlib
from tqdm import tqdm

import text_to_image

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_images', dest='num_images', type=int, default=4, help='Number of images to generate per prompt')
    parser.add_argument('--checkpoint_path', dest='model_path', type=str, default='./input/model/sd-v1-4.ckpt', help='Path to the model')
    parser.add_argument('--prompts_file', type=str, default='./input/prompts.txt', help='Path to the file containing the prompts, each on a line')
    parser.add_argument('--output', type=str, default='./outputs', help='Path to the output directory')
    args = parser.parse_args()


    # Load the diffusion model
    model_path = args.model_path

    # Load the prompts
    prompts_path = args.prompts_file
    if not os.path.exists(prompts_path):
        print(f'Error: prompts file {prompts_path} does not exist')
        return
    

    with open(prompts_path, 'r') as f:
        prompts = f.readlines()
    
    print("INFO: This script will generate a total of %s images" % (args.num_images * len(prompts)))

    text2img = text_to_image.Txt2Img(checkpoint_path=model_path,)
    for (index, prompt) in enumerate(prompts):
        print("\nINFO: Currently generating %s images for prompt \"%s\"" % (args.num_images, prompt))
        print("Progress: %s/%s (%.2f%%) (from total prompts)" % (index, len(prompts), (index/len(prompts))*100))

        hash = hashlib.sha1(prompt.encode("UTF-8")).hexdigest()
        filename = hash[:10] + ".jpg"
        dest_path = os.path.join(args.output, filename)

        prompt = prompt.strip()

        if not os.path.isfile(dest_path):
            text2img(
                dest_path=dest_path,
                prompt=prompt,
                batch_size=args.num_images,
            )
        

if __name__ == '__main__':
    main()
