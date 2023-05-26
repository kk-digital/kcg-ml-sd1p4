# DataUtility folder

## What is this?

This folder contains the scripts that are used to generate, download and manage the data used in the project.

## Summary

- [DataUtility folder](#datautility-folder)
  - [What is this?](#what-is-this)
  - [Summary](#summary)
  - [How to use the scripts?](#how-to-use-the-scripts)
    - [download\_model.py](#download_modelpy)
    - [generate\_noise\_images.py](#generate_noise_imagespy)
    - [image\_from\_prompt\_list.py](#image_from_prompt_listpy)
    - [image\_to\_image.py](#image_to_imagepy)
    - [inpaint\_image.py](#inpaint_imagepy)
    - [text\_to\_image.py](#text_to_imagepy)

## How to use the scripts?

OBS: The scripts must be run from the root folder of the project. Otherwise, the script will not work.

### download_model.py

**Command line arguments:**

- `--list-models`: List all the available models to download, when this argument is used, the script will ignore all other arguments and only list the models.
- `--model`: Name of the model to download, this argument is required when `--list-models` is not used.
- `--output`: Destination to save the model to, this argument is optional and defaults to _`/tmp/input/models/`_.

**Example usage:**

This command will list all the available models to download:
```bash
python3 download_model.py --list-models
```

This command will download the model `model_1` to `/tmp/input/models/`:
```bash
python3 download_model.py --model model_1
```

This command will download the model `model_1` to `/tmp/input/models/`:
```bash
python3 download_model.py --model model_1 --output /tmp/input/models/
```

### generate_noise_images.py

**Command line arguments:**

- `--prompt_prefix`: Prefix for the prompt, must end with "in the style of". Default value="A woman with flowers in her hair in a courtyard, in the style of".
- `--artist_file`:  Path to the file containing the artists, each on a line. Defaults to _'../input/prompts/artists.txt'_.
- `--output`: Path to the output directory. Defaults to _'./outputs'_.
- `--checkpoint_path`: Path to the checkpoint file. Defaults to _'./input/model/sd-v1-4.ckpt'_.
- `--sampler`: Name of the sampler to use. Defaults to _'ddim'. Options are 'ddim' and 'ddpm'_.
- `--steps`: Number of steps to use. Defaults to _20_.
- `--num_seeds`: Number of seeds to use. Defaults to _8_.
- `--noise_file`: Path to the file containing the noise seeds, each on a line. Defaults to _'noise-seeds.txt'_.

**Example usage:**

This command will generate 8 images for each of the 10 artists in the file `artists.txt`:
```bash
python3 generate_noise_images.py
```

This command will generate 8 images for each of the 10 artists in the file `artists.txt`:
```bash
python3 generate_noise_images.py --output ./outputs --checkpoint_path ./input/model/sd-v1-4.ckpt --sampler_name ddim --steps 20 --num_seeds 8 --noise_file noise-seeds.txt
```

### image_from_prompt_list.py

**Command line arguments:**

- `--num_images`: Number of images to generate per prompt. Defaults to _4_.
- `--checkpoint_path`: Path to the model. Defaults to _'./input/model/sd-v1-4.ckpt'_.
- `--prompts_file`: Path to the file containing the prompts, each on a line. Defaults to _'./input/prompts.txt'_.
- `--output`: Path to the output directory. Defaults to _'./outputs'_.

**Example usage:**

This command will generate 4 images for each of the 10 prompts in the file `prompts.txt`:
```bash
python3 image_from_prompt_list.py
```

This command will generate 4 images for each of the 10 prompts in the file `prompts.txt`:
```bash
python3 image_from_prompt_list.py --num_images 4 --checkpoint_path ./input/model/sd-v1-4.ckpt --prompts_file prompts.txt --output ./outputs
```

### image_to_image.py

**Command line arguments:**

- `--prompt`: Prompt to generate the image from. Defaults to _'a painting of a cute monkey playing guitar'_.
- `--orig_img`: Path to the original image. MANDATORY.
- `--batch_size`: Batch size to use. Defaults to _4_.
- `--steps`: Number of steps to use. Defaults to _50_.
- `--scale`: Unconditional guidance scale. Defaults to _5.0_.
- `--strength`: Strength for noise. Defaults to _0.75_.
- `--checkpoint_path`: Path to the checkpoint file. Defaults to _'./input/model/sd-v1-4.ckpt'_.
- `--output`: Path to the output directory. Defaults to _'./outputs'_.
- `--force_cpu`: Force the use of CPU. Defaults to _False_.
- `--cuda-device`: CUDA device to use. Defaults to _cuda:0_.

**Example usage:**

This command will update an image from the prompt `a painting of a cute monkey playing guitar`:
```bash
python3 image_from_prompt.py --orig_img ./input/orig-images/monkey.jpg
```

This command will update an image from the prompt `a painting of a cute monkey playing guitar`, using CPU with a low number of steps:
```bash
python3 image_from_prompt.py --orig_img ./input/orig-images/monkey.jpg --force_cpu --steps 10
```

### inpaint_image.py

**Command line arguments:**

- `--prompt`: Prompt to generate the image from. Defaults to _'a painting of a cute monkey playing guitar'_.
- `--checkpoint_path`: Path to the checkpoint file. Defaults to _'./input/model/sd-v1-4.ckpt'_.
- `--orig_img`: Path to the input image. MANDATORY.
- `--output`: Path to the output directory. Defaults to _'./outputs'_.
- `--batch_size`: Batch size to use. Defaults to _4_.
- `--steps`: Number of steps to use. Defaults to _50_.
- `--scale`: Unconditional guidance scale. Defaults to _5.0_.
- `--strength`: Strength for noise. Defaults to _0.75_.
- `--force_cpu`: Force the use of CPU. Defaults to _False_.
- `--cuda-device`: CUDA device to use. Defaults to _cuda:0_.

**Example usage:**

This command will inpaint an image from the prompt `a painting of a cute monkey playing guitar`:
```bash
python3 inpaint_image.py --orig_img ./input/orig-images/monkey.jpg
```

This command will inpaint an image from the prompt `a painting of a cute monkey playing guitar`, using CPU with a low number of steps:
```bash
python3 inpaint_image.py --orig_img ./input/orig-images/monkey.jpg --force_cpu --steps 10
```

### text_to_image.py

**Command line arguments:**

- `--prompt`: Prompt to generate the image from. Defaults to _'a painting of a cute monkey playing guitar'_.
- `--batch_size`: Batch size to use. Defaults to _4_.
- `--output`: Output path to store the generated images. Defaults to _'./outputs'_.
- `--sampler`: Set the sampler. Defaults to _'ddim'_. Options are _'ddim'_ and _'ddpm'_.
- `--checkpoint_path`: Relative path of the checkpoint file (*.ckpt). Defaults to _'./sd-v1-4.ckpt'_.
- `--flash`: Whether to use flash attention. Defaults to _False_.
- `--steps`: Number of sampling steps. Defaults to _50_.
- `--scale`: Unconditional guidance scale: **_eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))_** Defaults to _5.0_.
- `--low_vram`: Reduce VRAM usage. Defaults to _False_.
- `--force_cpu`: Force the use of CPU. Defaults to _False_.
- `--cuda-device`: CUDA device to use. Defaults to _cuda:0_.
