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


## Preparing Image Dataset
#### Validate Image Dataset
```
usage: process_dataset.py dataset-validate [-h] [--image-dataset-path IMAGE_DATASET_PATH] [--is_tagged IS_TAGGED] [--tmp-path TMP_PATH]

CLI tool for validating image dataset storage format

options:
  -h, --help            show this help message and exit
  --image-dataset-path IMAGE_DATASET_PATH
                        The path of the image dataset to validate
  --is-tagged IS_TAGGED
                        True if dataset is a tagged dataset (default=False)
```

#### Format and Compute Manifest Of Image Dataset
```
usage: process_dataset.py format-and-compute-manifest [-h] [--image-dataset-path IMAGE_DATASET_PATH] [--output-path OUTPUT_PATH] [--is-tagged IS_TAGGED] [--tmp-path TMP_PATH]

CLI tool for formatting images and computing manifest of dataset

options:
  -h, --help            show this help message and exit
  --image-dataset-path IMAGE_DATASET_PATH
                        The path of the image dataset to process
  --is-tagged IS_TAGGED
                        True if dataset is a tagged dataset (default=False)
  --output-path OUTPUT_PATH   The path to save the dataset (default='./output')

```

Sample command for untagged dataset:

``` shell
python3 ./scripts/process_dataset.py format-and-compute-manifest --image-dataset-path "./input/set_0000.zip" --output-path="./output"
```

#### Compute Features Of Zip Dataset
```
usage: process_dataset.py compute-features-of-zip [-h] [--image-dataset-path IMAGE_DATASET_PATH] [--clip-model CLIP_MODEL] [--batch-size BATCH_SIZE] [--output-path OUTPUT_PATH] [--tmp-path TMP_PATH]

CLI tool for computing features of dataset

options:
  -h, --help            show this help message and exit
  --image-dataset-path IMAGE_DATASET_PATH
                        The path of the image dataset to process
  --clip-model CLIP_MODEL
                        The CLIP model to use for computing features (default='ViT-L/14')
  --batch-size BATCH_SIZE
                        The batch size to use (default=8)
```

Sample command for tagged/untagged dataset: 

``` shell
python3 ./scripts/process_dataset.py compute-features-of-zip --image-dataset-path "./input/set_0000.zip" --clip-model="ViT-L/14"
```

## Sort The dataset
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
python3 ./scripts/chad_sort.py --dataset-path "input/set_0000_features.zip" --output-path "./output/chad_sort/"
```