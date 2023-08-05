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

## Running Stable Diffusion scripts

### Text To Image

Takes in a prompt and generates images

These are the available CLI arguments:

```
options:
  --prompt        PROMPT
                        An array of strings to help guide the generation process
  --batch_size   BATCH_SIZE
                        How many images to generate at once
  --output        OUTPUT
                        Number of folders to
  --sampler       SAMPLER
                        Name of the sampler to use
  --checkpoint_path     CHECKPOINT_PATH
                        Path to the checkpoint file
  --flash         FLASH
                        Whether to use flash attention
  --steps         STEPS
                        Number of steps to use
  --cgf_scale         SCALE
                        Unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
  --seed      SEED
                        Array of seed for the image generation: example '0, 1, 0, 7', Its better if the size of the array is the same as the number of generated images
  --low-vram      LOW_VRAM
                        Limit vram usage
  --force_cpu     FORCE_CPU
                        Force cpu usage
  --cuda_device   CUDA_DEVICE
                        Cuda device to use for generation process
  --num_images    NUM_IMAGES
                        Number of images to output
```

Example Usage:
``` shell
python3 ./scripts/text_to_image.py --prompt "character, chibi, waifu, side scrolling, white background, centered" --checkpoint_path "./input/model/v1-5-pruned-emaonly.safetensors" --batch_size 1 --num_images 6
```

### Random Prompts Generation and Disturbing Embeddings Image Generation

Try running:

```bash
python3 ./scripts/data_bounding_box_and_score_and_embedding_dataset.py --num_iterations 10 
```

- `cfg_strength`: Configuration strength. Defaults to `12`.
- `embedded_prompts_dir`: The path to the directory containing the embedded prompts tensors. Defaults to a constant EMBEDDED_PROMPTS_DIR, which is expected to be `'./input/embedded_prompts/'`
- `num_iterations`: The number of iterations to batch-generate images. Defaults to `8`.
- `batch_size`: The number of images to generate per batch. Defaults to `1`.
- `seed`: The noise seed used to generate the images. Defaults to random int `0 to 2^24``.
- `noise_multiplier`: The multiplier for the amount of noise used to disturb the prompt embedding. Defaults to `0.008`.
- `cuda_device`: The CUDA device to use. Defaults to `'get_device()'`.
- `clear_output_dir`: If True, the output directory will be cleared before generating images. Defaults to `False`.

### Generate Images Random Prompt

To generate images from random prompt, these are the available CLI arguments:

```
options:
  --batch_size   BATCH_SIZE
                        How many images to generate at once
  --output        OUTPUT
                        Number of folders to
  --sampler       SAMPLER
                        Name of the sampler to use
  --checkpoint_path     CHECKPOINT_PATH
                        Path to the checkpoint file
  --flash         FLASH
                        Whether to use flash attention
  --steps         STEPS
                        Number of steps to use
  --cgf_scale         SCALE
                        Unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
  --seed      SEED
                        Array of seed for the image generation: example '0, 1, 0, 7', Its better if the size of the array is the same as the number of generated images
  --low-vram      LOW_VRAM
                        Limit vram usage
  --force_cpu     FORCE_CPU
                        Force cpu usage
  --cuda_device   CUDA_DEVICE
                        Cuda device to use for generation process
  --num_images    NUM_IMAGES
                        Number of images to output
  --image_width IMAGE_WIDTH
                        Generated image width, default is 512
  --image_height IMAGE_HEIGHT
                        Generated image width, default is 512
```

``` shell
python3 ./scripts/generate_images_from_random_prompt.py --checkpoint_path "./input/model/v1-5-pruned-emaonly.safetensors" --cfg_scale 7 --num_images 10 --output "/output/"
```

