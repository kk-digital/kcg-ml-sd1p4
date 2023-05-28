import os
import hashlib

import text_to_image

from stable_diffusion.utils.model import save_images

from cli_builder import CLI

def main():
    args = CLI('Generate images from a list of prompts') \
        .batch_size() \
        .checkpoint_path() \
        .prompts_file() \
        .output() \
        .parse()

    # Load the diffusion model
    model_path = args.checkpoint_path

    # Load the prompts
    prompts_path = args.prompts_file
    with open(prompts_path, 'r') as f:
        prompts = f.readlines()
    
    print("INFO: This script will generate a total of %s images" % (args.batch_size * len(prompts)))

    text2img = text_to_image.Txt2Img(checkpoint_path=model_path,)
    text2img.initialize_script()
    
    for (index, prompt) in enumerate(prompts):
        print("\nINFO: Currently generating %s images for prompt \"%s\"" % (args.batch_size, prompt))
        print("Progress: %s/%s (%.2f%%) (from total prompts)" % (index, len(prompts), (index/len(prompts))*100))

        hash = hashlib.sha1(prompt.encode("UTF-8")).hexdigest()
        filename = hash[:10] + ".jpg"
        dest_path = os.path.join(args.output, filename)

        prompt = prompt.strip()

        if not os.path.isfile(dest_path):
            images = text2img.generate_images(
                prompt=prompt,
                batch_size=args.batch_size,
            )

            save_images(images, dest_path)
        

if __name__ == '__main__':
    main()
