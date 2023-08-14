# Dataset Generation

## Prerequisites

### Installing Requirements
Run: 
```bash
pip3 install -r requirements.txt
```

### Downloading Models
Run: 
```bash
python3 ./download_models.py
```

### Processing Models
Run: 
```bash
python3 ./process_models.py
```

## Generating A Dataset

### Generate Images Random Prompt

To generate images from random prompt, these are the available CLI arguments:

```
options:
  -h, --help            show this help message and exit
  --prompts_file PROMPTS_FILE
                        Path to the file containing the prompts, each on a line (default: './input/prompts.txt')
  --batch_size BATCH_SIZE
                        How many images to generate at once (default: 1)
  --output OUTPUT       Path to the output directory (default: /output)
  --sampler SAMPLER     Name of the sampler to use (default: ddim)
  --checkpoint_path CHECKPOINT_PATH
                        Path to the checkpoint file (default: './input/model/v1-5-pruned-emaonly.safetensors')
  --flash               whether to use flash attention
  --steps STEPS         Number of steps to use (default: 20)
  --cfg_scale CFG_SCALE
                        unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
  --low_vram            limit VRAM usage
  --force_cpu           force CPU usage
  --cuda_device CUDA_DEVICE
                        cuda device to use for generation
  --num_images NUM_IMAGES
                        How many images to generate (default: 1)
  --seed SEED           Seed for the image generation (default: )
  --output_metadata     outputs the metadata
  --image_width IMAGE_WIDTH
                        Generate image width (default: 512)
  --image_height IMAGE_HEIGHT
                        Generate image height (default: 512)
  --num_datasets NUM_DATASETS
                        Number of datasets to generate (default: 1)
```

``` shell
python3 ./scripts/generate_images_from_random_prompt.py --checkpoint_path "./input/model/v1-5-pruned-emaonly.safetensors" --cfg_scale 7 --num_images 10 --output "/output/"
```

### Generate Images From Prompt Generator

To generate images from random prompt, these are the available CLI arguments:

```
options:
  -h, --help            show this help message and exit
  --prompts_file PROMPTS_FILE
                        Path to the file containing the prompts, each on a line (default: './input/prompts.txt')
  --batch_size BATCH_SIZE
                        How many images to generate at once (default: 1)
  --output OUTPUT       Path to the output directory (default: /output)
  --sampler SAMPLER     Name of the sampler to use (default: ddim)
  --checkpoint_path CHECKPOINT_PATH
                        Path to the checkpoint file (default: './input/model/v1-5-pruned-emaonly.safetensors')
  --flash               whether to use flash attention
  --steps STEPS         Number of steps to use (default: 20)
  --cfg_scale CFG_SCALE
                        unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
  --low_vram            limit VRAM usage
  --force_cpu           force CPU usage
  --cuda_device CUDA_DEVICE
                        cuda device to use for generation
  --num_images NUM_IMAGES
                        How many images to generate (default: 1)
  --seed SEED           Seed for the image generation (default: )
  --output_metadata     outputs the metadata
  --image_width IMAGE_WIDTH
                        Generate image width (default: 512)
  --image_height IMAGE_HEIGHT
                        Generate image height (default: 512)
  --num_datasets NUM_DATASETS
                        Number of datasets to generate (default: 1)
  --num_phrases [NUM_PHRASES]
                        The number of phrases for the prompt to generate
```

``` shell
python3 ./scripts/generate_images_from_prompt_generator.py --checkpoint_path "./input/model/v1-5-pruned-emaonly.safetensors" --cfg_scale 7 --num_images 10 --num_phrases 12 --output "./output/"
```

## Sort The dataset