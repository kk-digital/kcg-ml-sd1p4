# kcg-ml-sd1p4

[![Build Status](http://103.230.15.156:8111/app/rest/builds/buildType:(id:KcgMlSd1p4_Build)/statusIcon)](http://teamcity/viewType.html?buildTypeId=KcgMlSd1p4_Build&guest=1)

## Summary

- [kcg-ml-sd1p4](#kcg-ml-sd1p4)
  - [Prerequisites](#prerequisites)
    - [Installing Requirements](#installing-requirements)
    - [Downloading Models](#downloading-models)
    - [Processing Models](#processing-models)
  - [Running Stable Diffusion scripts](#running-stable-diffusion-scripts)
    - [Text To Image](#text-to-image)
    - [Embed prompts](#embed-prompts)
    - [Images from embeddings](#images-from-embeddings)
    - [Images from distributions](#images-from-distributions)
    - [Images from temperature range](#images-from-temperature-range)
    - [Images and encodings](#images-and-encodings)
    - [Perturbations on prompts embeddings](#perturbations-on-prompts-embeddings)
    - [Random Prompts Generation and Disturbing Embeddings Image Generation](#random-prompts-generation-and-disturbing-embeddings-image-generation)
    - [Image Grid Generator](#image-grid-generator)
    - [Generate Images Random Prompt](#generate-images-random-prompt)
    - [Generate Images From Prompt Generator](#generate-images-from-prompt-generator)
    - [Chad Score](#chad-score)
    - [Chad Sort](#chad-sort)
    - [Running GenerationTask](#running-generationtask)
    - [Prompt Score](#prompt-score)
    - [Prompt Embeddings Gradient Optimization](#prompt-embeddings-gradient-optimization)
    - [Prompt Generator](#prompt-generator)
    - [Auto-ml](#auto-ml)
    - [Generate Images From Model Predictions](#generate-images-from-model-predictions)
  - [Tests](#tests)
    - [To run all scripts](#to-run-all-scripts)
    - [To run all kcg-ml-sd1p4 tests](#to-run-all-kcg-ml-sd1p4-tests)

## Prerequisites

### Installing Requirements
Run: 
```bash
pip3 install -r requirements.txt
```

### Downloading Models

This script will download these models:

    CLIP - ./input/model/clip/vit-large-patch14/model.safetensors
    CLIP Text model - ./input/model/clip/text_model/pytorch_model.bin
    Stable Diffusion - ./input/model/sd/v1-5-pruned-emaonly/v1-5-pruned-emaonly.safetensors

Run: 
```bash
python3 ./download_models.py
```

### Processing Models
_Note: This script takes in a stable diffusion model and cuts it up into sub-models_

Run: 
```bash
python3 ./process_models.py
```

Then you already should be able to run:

```bash
python3 ./scripts/txt2img.py --num_images 2 --prompt 'A purple rainbow, filled with grass'
```

## Running Stable Diffusion scripts

### Text To Image

Takes in a prompt and generates images

These are the available CLI arguments:

```
options:
  -h, --help            show this help message and exit
  --prompt [PROMPT]     The prompt to render
  --negative-prompt [NEGATIVE_PROMPT]
                        The negative prompt. For things we dont want to see in generated image
  --prompts_file PROMPTS_FILE
                        Path to the file containing the prompts, each on a line (default: './input/prompts.txt')
  --batch_size BATCH_SIZE
                        How many images to generate at once (default: 1)
  --output OUTPUT       Path to the output directory (default: ./output)
  --sampler SAMPLER     Name of the sampler to use (default: ddim)
  --checkpoint_path CHECKPOINT_PATH
                        Path to the checkpoint file (default: './input/model/v1-5-pruned-emaonly.safetensors')
  --flash               whether to use flash attention
  --steps STEPS         Number of steps to use (default: 50)
  --cfg_scale CFG_SCALE
                        unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
  --low_vram            limit VRAM usage
  --force_cpu           force CPU usage
  --cuda_device CUDA_DEVICE
                        cuda device to use for generation
  --num_images NUM_IMAGES
                        How many images to generate (default: 1)
  --seed SEED           Seed for the image generation (default: )
```

Example Usage:
``` shell
python3 ./scripts/text_to_image.py --prompt "character, chibi, waifu, side scrolling, white background, centered" --negative-prompt "white" --checkpoint_path "./input/model/v1-5-pruned-emaonly.safetensors" --batch_size 1 --num_images 1
```

### Embed prompts

Saves a tensor of a batch of prompts embeddings, and the tensor for the null prompt `""`.

Command:

```bash
python3 ./scripts/embed_prompts.py --prompts 'A painting of a computer virus', 'An old photo of a computer scientist', 'A computer drawing a computer'
```

**Command line arguments**

- `-p, --prompts`: The prompts to embed. Defaults to `['A painting of a computer virus', 'An old photo of a computer scientist']`.
- `--embedded_prompts_dir`: The path to the directory containing the embedded prompts tensors. Defaults to a constant `EMBEDDED_PROMPTS_DIR`, which is expected to be `'./input/embedded_prompts/'`.
-
### Images from embeddings

Only run this _after_ generating the embedded prompts with the [above script](#embed-prompts).


**Command line arguments**

- `-p, --embedded_prompts_dir`: The path to the directory containing the embedded prompts tensors. Defaults to the `EMBEDDED_PROMPTS_DIR` constant, which is expected to be `'./input/embedded_prompts/'`.
- `-od, --output_dir`: The output directory. Defaults to the `OUTPUT_DIR` constant, which is expected to be `'./output/noise-tests/from_embeddings'`.
- `--num_images`: Number of images to generate. Defaults to `1`.
- `-bs, --batch_size`: Batch size to use. Defaults to `1`.
- `-t, --temperature`: Sampling temperature. Defaults to `1.0`.
- `--ddim_eta`: Amount of noise to readd during the sampling process. Defaults to `0.0`.
- `--clear_output_dir`: Either to clear or not the output directory before running. Defaults to `False`.
- `--cuda_device`: CUDA device to use. Defaults to `get_device()`.
- `--low_vram`: Only use this if the gpu you have is old`.
- `--sampler`: Default value is `DDIM`.
- `--seed`: Default value is `empty string`.
- `--cfg_scale`: unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty)) . Defaults to `7`.
- 
Try running:

```bash
python3 ./scripts/generate_images_from_embeddings.py --num_images 4 --temperature 1.2 --ddim_eta 0.2
```

### Images from distributions

Try running:
```bash
python3 ./scripts/generate_images_from_distributions.py -d 4 --params_steps 4 --params_range 0.49 0.54 --num_seeds 4 --temperature 1.2 --ddim_eta 1.2
```

**Command line arguments**

- `-p, --prompt`: The prompt to generate images from. Defaults to `"A woman with flowers in her hair in a courtyard, in the style of Frank Frazetta"`.
- `--negative-prompt`: The negative prompt. For things we dont want to see in generated image. Defaults to `''`.
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

### Images from temperature range

Try running:

```bash
python3 ./scripts/generate_images_from_temperature_range.py -d 4 --params_range 0.49 0.54 --params_steps 3 --temperature_steps 3 --temperature_range 0.8 2.0
```

**Command line arguments**

- `-p, --prompt`: The prompt to generate images from. Defaults to `"A woman with flowers in her hair in a courtyard, in the style of Frank Frazetta"`.
- `--negative-prompt`: The negative prompt. For things we dont want to see in generated image. Defaults to `''`.
- `-od, --output_dir`: The output directory. Defaults to the `OUTPUT_DIR` constant, which is expected to be `"./output/noise-tests/temperature_range"`.
- `-cp, --checkpoint_path`: The path to the checkpoint file to load from. Defaults to the `CHECKPOINT_PATH` constant, which is expected to be `"./input/model/v1-5-pruned-emaonly.safetensors"`.
- `-F, --fully_initialize`: Whether to fully initialize or not. Defaults to `False`.
- `-d, --distribution_index`: The distribution index to use. Defaults to 4. Options: 0: "Normal", 1: "Cauchy", 2: "Gumbel", 3: "Laplace", 4: "Logistic".
- `-s, --seed`: The seed value. Defaults to random int `0 to 2^24`.
- `-bs, --batch_size`: The batch size to use. Defaults to `1`.
- `--params_steps`: The number of steps for the parameters. Defaults to `3`.
- `--params_range`: The range of parameters. Defaults to `[0.49, 0.54]`.
- `--temperature_steps`: The number of steps for the temperature. Defaults to `3`.
- `--temperature_range`: The range of temperature. Defaults to `[1.0, 4.0]`.
- `--ddim_eta`: The value of ddim_eta. Defaults to `0.1`.
- `--clear_output_dir`: Whether to clear the output directory or not. Defaults to `False`.
- `--cuda_device`: The CUDA device to use. Defaults to `"get_device()"`.


### Images and encodings

**Command line arguments**

- `--batch_size`: How many images to generate at once. Defaults to `1`.
- `--num_iterations`: How many times to iterate the generation of a batch of images. Defaults to `10`.
- `--prompt`: The prompt to render. It is an optional argument. Defaults to `"a painting of a cute monkey playing guitar"`.
- `--negative-prompt`: The negative prompt. For things we dont want to see in generated image. Defaults to `''`.
- `--cuda_device`: CUDA device to use for generation. Defaults to `"get_device()"`.
- `--low_vram`: Flag for low vram gpus. Defaults to `False`.
- `--sampler`: Sampler name. Defaults to `ddim`.
- `--cfg_scale`: Unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty)) . Defaults to `7`.
- `--seed`: Array of seed values, one for each generated image. Defaults to ``.


Try running:
```bash
python3 ./scripts/generate_images_and_encodings.py --prompt "An oil painting of a computer generated image of a geometric pattern" --num_iterations 10
```

### Perturbations on prompts embeddings

Try running:
```bash
python3 ./scripts/embed_prompts_and_generate_images.py
```
Outputs in: `./output/disturbing_embeddings`

- `--prompt`: The prompt to embed. Defaults to `"A woman with flowers in her hair in a courtyard, in the style of Frank Frazetta"`.
- `--num_iterations`: The number of iterations to batch-generate images. Defaults to `8`.
- `--seed`: The noise seed used to generate the images. Defaults to random int `0 to 2^24`.
- `--noise_multiplier`: The multiplier for the amount of noise used to disturb the prompt embedding. Defaults to `0.01`.
- `--cuda_device`: The CUDA device to use. Defaults to `"get_device()"`.


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


### Image Grid Generator

The Image Grid generator is a script that generates a grid of images from a directory or a zip file containing images. 

Run the script with the following command:

    python ./utility/scripts/grid_generator.py --input_path ./test/test_images/clip_segmentation --output_path ./tmp --rows 3 --columns 2  --img_size 256


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
  --num-phrases [NUM_PHRASES]
                        The number of phrases for the prompt to generate
```

``` shell
python3 ./scripts/generate_images_from_prompt_generator.py --checkpoint_path "./input/model/v1-5-pruned-emaonly.safetensors" --cfg_scale 7 --num_images 10 --num_phrases 12 --output "./output/"
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

Example Usage:
``` shell
python3 ./scripts/chad_score.py --model-path="input/model/chad_score/chad-score-v1.pth" --image-path="test/test_images/test_img.jpg"
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

Example Usage:

``` shell
python3 ./scripts/chad_sort.py --dataset-path "test/test_zip_files/test-dataset-correct-format.zip" --output-path "/output/chad_sort/"
```

### Running GenerationTask

Runs a generation task from .json file

These are the available CLI arguments:

```
options:
  --task_path TASK_PATH
                        Path to the task .json file
```

Example Usage:
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
  --output_path OUTPUT_PATH
                        Path output folder
  --model_output_name MODEL_OUTPUT_NAME
                        Filename of the trained model
  --num_epochs NUM_EPOCHS
                        Number of epochs (default: 1000)
  --epsilon_raw EPSILON_RAW
                        Epsilon for raw data (default: 10.0)
  --epsilon_scaled EPSILON_SCALED
                        Epsilon for scaled data (default: 0.2)
  --use_76th_embedding  If this option is set, only use the last entry in the embeddings tensor
  --show_validation_loss
                        whether to show validation loss
```

Example Usage:
``` shell
python scripts/prompt_score.py --input_path input/set_0000_v2.zip --use_76th_embedding --num_epochs 200 --epsilon_raw 10 --epsilon_scaled 0.2 --model_output_name prompt_score.pth
```

### Prompt Embeddings Gradient Optimization

Optimizes an embedding vector using gradients.

``` shell
options:
  --input_path INPUT_PATH
                        Path to input zip
  --model_path MODEL_PATH
                        Path to the model
  --iterations ITERATIONS
                        How many iterations to perform
  --learning_rate LEARNING_RATE
                        Learning rate to use when optimizing
```

Example Usage:

``` shell
python scripts/prompt_gradient.py --input_path input/set_0000_v2.zip --model_path output/models/prompt_score.pth --iterations 10 --learning_rate 0.01
```

### Prompt Generator
Generates prompts and saves to a json file
```
usage: prompt_generator.py [-h] [--num-prompts NUM_PROMPTS] [--num-phrases NUM_PHRASES] [--output OUTPUT]

Prompt Generator CLI tool

options:
  -h, --help            show this help message and exit
  --num-prompts NUM_PROMPTS
                        Number of prompts to generate
  --num-phrases NUM_PHRASES
                        Number of phrases per prompt
  --output OUTPUT       Output path for prompt list json

```

Example Usage:
```
python ./scripts/prompt_generator.py --num-prompts 1000 --num-phrases 12 --output "./output/prompt_list_1000.json" 
```

### Auto-ml
_Note: Current only support dataset generated from Generate Images Random Prompt_
```
usage: auto_ml.py [-h] [--x-input X_INPUT] [--total-time TOTAL_TIME] [--per-run-time PER_RUN_TIME] [--dataset-zip-path DATASET_ZIP_PATH] [--output OUTPUT]

Run automl on a dataset with embeddings or clip feature and chad score

options:
  -h, --help            show this help message and exit
  --x-input X_INPUT     X-input will be either embedding or clip
  --total-time TOTAL_TIME
                        Time limit in seconds for the search of appropriate models
  --per-run-time PER_RUN_TIME
                        Time limit for a single call to the machine learning model
  --dataset-zip-path DATASET_ZIP_PATH
                        Path to the dataset to be used
  --output OUTPUT       Output path where the plot image will be saved
```
Example Usage:
``` shell
python scripts/auto_ml.py --dataset-zip-path ./input/set_0002.zip --x-input "clip" --output "./output"
```

### Generate Images From Model Predictions
_Note: Currently only supports prediction output from kcg-ml/elm-regression_
```
usage: generate_images_from_predictions.py [-h] [--dataset-path DATASET_PATH] [--output-path OUTPUT_PATH] [--num-class NUM_CLASS] [--limit-per-class LIMIT_PER_CLASS]

Generate Image For Each Range Of Predicted Chad Score

options:
  -h, --help            show this help message and exit
  --dataset-path DATASET_PATH
                        Predictions json path
  --output-path OUTPUT_PATH
                        Output path for the generated images
  --num-class NUM_CLASS
                        Number of classes to sort the images
  --limit-per-class LIMIT_PER_CLASS
                        Number of images to generate per class
```

Example Usage:
```
python ./scripts/generate_images_from_predictions.py --dataset-path "./output/chad-score-prediction.json" --output-path "./output/sort-generate-test" --num-class 10 --limit-per-class 1 
```
## Tests
### To run all scripts
1. Make an env

    `python3 -m venv env`
2. Activate env

    `source env/bin/activate`
3. Install requirements

    `pip install -r requirements.txt`
4. run pytest

    `python -m unittest test_main_scripts.py `

### To run all kcg-ml-sd1p4 tests
1. Make an env

    `python3 -m venv env`
2. Activate env

    `source env/bin/activate`
3. Install requirements

    `pip install -r requirements.txt`
4. run pytest

    `pytest ./test/test_scripts"`