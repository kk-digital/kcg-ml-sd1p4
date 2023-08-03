# kcg-ml-sd1p4

[![Build Status](http://85.17.25.95:8111/app/rest/builds/buildType:(id:KcgMlSd1p4_Build)/statusIcon)](http://teamcity/viewType.html?buildTypeId=KcgMlSd1p4_Build&guest=1)

## Summary

- [kcg-ml-sd1p4](#kcg-ml-sd1p4)
  - [Summary](#summary)
  - [Installing requirements](#installing-requirements)
  - [Downloading models and setting up directories](#downloading-models-and-setting-up-directories)
    - [Text To Image](#text-to-image)
      - [Example Usage:](#example-usage)
  - [Running `stable_diffusion` scripts](#running-stable_diffusion-scripts)
      - [Embed prompts](#embed-prompts)
      - [Images from embeddings](#images-from-embeddings)
      - [Images from distributions](#images-from-distributions)
      - [Images from temperature range](#images-from-temperature-range)
      - [Images and encodings](#images-and-encodings)
      - [Perturbations on prompts embeddings](#perturbations-on-prompts-embeddings)
      - [Random Pormpts Generation and Disturbing Embeddings Image Generation](#random-pormpts-generation-and-disturbing-embeddings-image-generation)
      - [Image Grid Generator](#image-grid-generator)
        - [Usage](#usage)
    - [Generate Images Random Prompt](#generate-images-random-prompt)
    - [Chad Score](#chad-score)
      - [Example Usage:](#example-usage-1)
      - [using python3](#using-python3)
      - [using python](#using-python)
    - [Chad Sort](#chad-sort)
      - [Example Usage:](#example-usage-2)
      - [using python3](#using-python3-1)
      - [using python](#using-python-1)
    - [Running GenerationTask](#running-generationtask)
      - [Example Usage:](#example-usage-3)
      - [using python3](#using-python3-2)
      - [using python](#using-python-2)

## Installing requirements
```bash
pip3 install -r requirements.txt
```

## Downloading models and setting up directories

We need two checkpoints: 

`v1-5-pruned-emaonly.safetensors` from `https://huggingface.co/runwayml/
stable-diffusion-v1-5/`; 

the safetensors version of `openai/clip-vit-large-patch14`, found in [refs/pr/19](https://huggingface.co/openai/clip-vit-large-patch14/tree/refs%2Fpr%2F19).

The `setup_directories_and_models.py` script will set up the folder structure, download both these models (depending on the args given), and process them.

```bash
python3 ./setup_directories_and_models.py --download_base_clip_model True --download_base_sd_model True
```

Then you already should be able to run:

```bash
python3 ./scripts/txt2img.py --num_images 2 --prompt 'A purple rainbow, filled with grass'
```
<!-- 
Download the currently supported checkpoint, `v1-5-pruned-emaonly.safetensors`, to `./input/model/`.

```bash
./download-model.sh
```

Alternatively, if you prefer to download directly, use [this link.](https://mega.nz/file/AVZnGbzL#EfXN4YINe0Wb7ukiqpCPa7THssugyCQU8pvpMpvxPAw)
## Saving submodels

Start by running this.

This script initializes a `LatentDiffusion` module, load its weights from a checkpoint file at `./input/model/v1-5-pruned-emaonly.safetensors` and then save all submodules.
**Note**: saving these models will take an extra ~5GB of storage.

```bash
python scripts/process_base_model.py input/model/v1-5-pruned-emaonly.safetensors
```

**Command line arguments**

- `-g, --granularity`: Determines the height in the model tree on which to save the submodels. Defaults to `0`, saving all submodels of the `LatentDiffusion` class and all submodels thereof. If `1` is given, it saves only up to the first level, i.e., autoencoder, UNet and CLIP text embedder. *Note*: that the other scripts and notebooks expect this to be ran with `0`.
- `--root_models_path`: Base directory for the models' folder structure. Defaults to the constant `ROOT_MODELS_DIR`, which is expected to be `./input/model/`.
- `--checkpoint_path`: Relative path of the checkpoint file (*.safetensors). Defaults to the constant `CHECKPOINT_PATH`, which is expected to be `'./input/model/v1-5-pruned-emaonly.safetensors' = os.path.join(ROOT_MODELS_DIR, 'v1-5-pruned-emaonly.safetensors')`. -->

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

#### Example Usage:

``` shell
python3 ./scripts/text_to_image.py --prompt "character, chibi, waifu, side scrolling, white background, centered" --checkpoint_path "./input/model/v1-5-pruned-emaonly.safetensors" --batch_size 1 --num_images 6
```

## Running `stable_diffusion` scripts

There are five other new scripts besides `scripts/process_base_model.py`.

#### Embed prompts

Saves a tensor of a batch of prompts embeddings, and the tensor for the null prompt `""`.

Command:

```bash
python3 ./scripts/embed_prompts.py --prompts 'A painting of a computer virus', 'An old photo of a computer scientist', 'A computer drawing a computer'
```

**Command line arguments**

- `-p, --prompts`: The prompts to embed. Defaults to `['A painting of a computer virus', 'An old photo of a computer scientist']`.
- `--embedded_prompts_dir`: The path to the directory containing the embedded prompts tensors. Defaults to a constant `EMBEDDED_PROMPTS_DIR`, which is expected to be `'./input/embedded_prompts/'`.
-
#### Images from embeddings

Only run this _after_ generating the embedded prompts with the [above script](#embed-prompts).

Try running:

```bash
python3 ./scripts/generate_images_from_embeddings.py --num_seeds 4 --temperature 1.2 --ddim_eta 0.2
```

**Command line arguments**

- `-p, --embedded_prompts_dir`: The path to the directory containing the embedded prompts tensors. Defaults to the `EMBEDDED_PROMPTS_DIR` constant, which is expected to be `'./input/embedded_prompts/'`.
- `-od, --output_dir`: The output directory. Defaults to the `OUTPUT_DIR` constant, which is expected to be `'./output/noise-tests/from_embeddings'`.
- `--num_seeds`: Number of random seeds to use. Defaults to `3`. Ranges from `1` to `7`.
- `-bs, --batch_size`: Batch size to use. Defaults to `1`.
- `-t, --temperature`: Sampling temperature. Defaults to `1.0`.
- `--ddim_eta`: Amount of noise to readd during the sampling process. Defaults to `0.0`.
- `--clear_output_dir`: Either to clear or not the output directory before running. Defaults to `False`.
- `--cuda_device`: CUDA device to use. Defaults to `get_device()`.

#### Images from distributions

Try running:
```bash
python3 ./scripts/generate_images_from_distributions.py -d 4 --params_steps 4 --params_range 0.49 0.54 --num_seeds 4 --temperature 1.2 --ddim_eta 1.2
```

**Command line arguments**

- `-p, --prompt`: The prompt to generate images from. Defaults to `"A woman with flowers in her hair in a courtyard, in the style of Frank Frazetta"`.
- `-od, --output_dir`: The output directory. Defaults to the `OUTPUT_DIR` constant, which should be `"./output/noise-tests/from_distributions"`.
- `-cp, --checkpoint_path`: The path to the checkpoint file to load from. Defaults to the `CHECKPOINT_PATH` constant, which should be `"./input/model/v1-5-pruned-emaonly.safetensors"`.
- `-F, --fully_initialize`: Whether to fully initialize or not. Defaults to `False`.
- `-d, --distribution_index`: The distribution index to use. Defaults to `4`. Options: 0: "Normal", 1: "Cauchy", 2: "Gumbel", 3: "Laplace", 4: "Logistic".
- `-bs, --batch_size`: The batch size to use. Defaults to `1`.
- `--params_steps`: The number of steps for the parameters. Defaults to `3`.
- `--params_range`: The range of parameters. Defaults to `[0.49, 0.54]`.
- `--num_seeds`: Number of random seeds to use. Defaults to `3`.
- `-t, --temperature`: Sampling temperature. Defaults to `1.0`.
- `--ddim_eta`: Amount of noise to readd during the sampling process. Defaults to `0.0`.
- `--clear_output_dir`: Either to clear or not the output directory before running. Defaults to `False`.
- `--cuda_device`: CUDA device to use. Defaults to `"get_device()"`.

#### Images from temperature range

Try running:

```bash
python3 ./scripts/generate_images_from_temperature_range.py -d 4 --params_range 0.49 0.54 --params_steps 3 --temperature_steps 3 --temperature_range 0.8 2.0
```

**Command line arguments**

- `-p, --prompt`: The prompt to generate images from. Defaults to `"A woman with flowers in her hair in a courtyard, in the style of Frank Frazetta"`.
- `-od, --output_dir`: The output directory. Defaults to the `OUTPUT_DIR` constant, which is expected to be `"./output/noise-tests/temperature_range"`.
- `-cp, --checkpoint_path`: The path to the checkpoint file to load from. Defaults to the `CHECKPOINT_PATH` constant, which is expected to be `"./input/model/v1-5-pruned-emaonly.safetensors"`.
- `-F, --fully_initialize`: Whether to fully initialize or not. Defaults to `False`.
- `-d, --distribution_index`: The distribution index to use. Defaults to 4. Options: 0: "Normal", 1: "Cauchy", 2: "Gumbel", 3: "Laplace", 4: "Logistic".
- `-s, --seed`: The seed value. Defaults to `2982`.
- `-bs, --batch_size`: The batch size to use. Defaults to `1`.
- `--params_steps`: The number of steps for the parameters. Defaults to `3`.
- `--params_range`: The range of parameters. Defaults to `[0.49, 0.54]`.
- `--temperature_steps`: The number of steps for the temperature. Defaults to `3`.
- `--temperature_range`: The range of temperature. Defaults to `[1.0, 4.0]`.
- `--ddim_eta`: The value of ddim_eta. Defaults to `0.1`.
- `--clear_output_dir`: Whether to clear the output directory or not. Defaults to `False`.
- `--cuda_device`: The CUDA device to use. Defaults to `"get_device()"`.


#### Images and encodings

Try running:
```bash
python3 ./scripts/generate_images_and_encodings.py --prompt "An oil painting of a computer generated image of a geometric pattern" --num_iterations 10
```

**Command line arguments**

- `--batch_size`: How many images to generate at once. Defaults to `1`.
- `--num_iterations`: How many times to iterate the generation of a batch of images. Defaults to `10`.
- `--prompt`: The prompt to render. It is an optional argument. Defaults to `"a painting of a cute monkey playing guitar"`.
- `--cuda_device`: CUDA device to use for generation. Defaults to `"get_device()"`.

#### Perturbations on prompts embeddings

Try running:
```bash
python3 ./scripts/embed_prompts_and_generate_images.py
```
Outputs in: `./output/disturbing_embeddings`

- `--prompt`: The prompt to embed. Defaults to `"A woman with flowers in her hair in a courtyard, in the style of Frank Frazetta"`.
- `--num_iterations`: The number of iterations to batch-generate images. Defaults to `8`.
- `--seed`: The noise seed used to generate the images. Defaults to `2982`.
- `--noise_multiplier`: The multiplier for the amount of noise used to disturb the prompt embedding. Defaults to `0.01`.
- `--cuda_device`: The CUDA device to use. Defaults to `"get_device()"`.


#### Random Prompts Generation and Disturbing Embeddings Image Generation

Try running:

```bash
python3 ./scripts/data_bounding_box_and_score_and_embedding_dataset.py --num_iterations 10 --save_embeddings True
```

- `--save_embeddings`: If True, the disturbed embeddings will be saved to disk. Defaults to `False`.
- `cfg_strength`: Configuration strength. Defaults to `12`.
- `embedded_prompts_dir`: The path to the directory containing the embedded prompts tensors. Defaults to a constant EMBEDDED_PROMPTS_DIR, which is expected to be `'./input/embedded_prompts/'`
- `num_iterations`: The number of iterations to batch-generate images. Defaults to `8`.
- `batch_size`: The number of images to generate per batch. Defaults to `1`.
- `seed`: The noise seed used to generate the images. Defaults to `2982`.
- `noise_multiplier`: The multiplier for the amount of noise used to disturb the prompt embedding. Defaults to `0.008`.
- `cuda_device`: The CUDA device to use. Defaults to `'get_device()'`.
- `clear_output_dir`: If True, the output directory will be cleared before generating images. Defaults to `False`.


#### Image Grid Generator

The Image Grid generator is a script that generates a grid of images from a directory or a zip file containing images. 

##### Usage

Run the script with the following command:

    python ./utility/scripts/grid_generator.py --input_path ./test/test_images/clip_segmentation --output_path ./tmp --rows 3 --columns 2  --img_size 256


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

### Chad Score

To get the chad score of an image, these are the available CLI arguments:

```
options:
  --image-path IMAGE_PATH
                        Path to the image to be scored
  --model-path MODEL_PATH
                        Path to the model used for scoring the image
```

#### Example Usage:

#### using python3
``` shell
python3 ./scripts/chad_score.py --model-path="input/model/chad_score/chad-score-v1.pth" --image-path="test/test_images/test_img.jpg"
```

#### using python
``` shell
python ./scripts/chad_score.py --model-path="input/model/chad_score/chad-score-v1.pth" --image-path="test/test_images/test_img.jpg"
```

### Chad Sort

Takes in an image dataset
Sorts the images in the dataset by chad score into multiple folders

These are the available CLI arguments:

```
options:
  --dataset-path DATASET_PATH
                        Path to the dataset to be sorted
  --output-path OUTPUT_PATH
                        Path to the output folder
  --num-classes NUM_CLASSES
                        Number of folders to output
```

#### Example Usage:

#### using python3
``` shell
python3 ./scripts/chad_sort.py --dataset-path "test/test_zip_files/test-dataset-correct-format.zip" --output-path "/output/chad_sort/"
```

#### using python
``` shell
python ./scripts/chad_sort.py --dataset-path "test/test_zip_files/test-dataset-correct-format.zip" --output-path "./output/chad_sort/"
```

### Running GenerationTask

Runs a generation task from .json file

These are the available CLI arguments:

```
options:
  --task_path TASK_PATH
                        Path to the task .json file
```

#### Example Usage:

#### using python3
``` shell
python3 ./scripts/run_generation_task.py --task_path './test/test_generation_task/text_to_image_v1.json'
```
``` shell
python3 ./scripts/run_generation_task.py --task_path './test/test_generation_task/generate_images_from_random_prompt_v1.json'
```

### Prompt Score

Creates a linear regression model that uses prompt embedding tensors as input,
and outputs a chad score.

``` shell
options:
--input_path INPUT_PATH
                        Path to input zip
  --num_epochs NUM_EPOCHS
                        Number of epochs (default: 1000)
  --epsilon_raw EPSILON_RAW
                        Epsilon for raw data (default: 10.0)
  --epsilon_scaled EPSILON_SCALED
                        Epsilon for scaled data (default: 0.2)
  --use_76th_embedding  If this option is set, only use the last entry in the embeddings tensor
  --show_validation_loss  If this option is set, shows validation loss during training
```

#### Example Usage:

``` shell
python scripts/prompt_score.py --input_path input/set_0000_v2.zip --use_76th_embedding --num_epochs 200 --epsilon_raw 10 --epsilon_scaled 0.2
```
