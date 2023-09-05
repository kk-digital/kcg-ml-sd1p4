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

### Generate Images From Prompt List Dataset

To generate images from prompt list dataset, these are the available CLI arguments:

```
options:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        How many images to generate at once (default: 1)
  --output OUTPUT       Path to the output directory (default: ./output)
  --sampler SAMPLER     Name of the sampler to use (default: ddim)
  --checkpoint_path CHECKPOINT_PATH
                        Path to the checkpoint file (default: './input/model/v1-5-pruned-emaonly.safetensors')
  --flash               whether to use flash attention
  --steps STEPS         Number of steps to use (default: 20)
  --cfg_scale CFG_SCALE
                        unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
  --force_cpu           force CPU usage
  --cuda_device CUDA_DEVICE
                        cuda device to use for generation
  --num_images NUM_IMAGES
                        How many images to generate (default: 1)
  --seed SEED           Seed for the image generation (default: )
  --image_width IMAGE_WIDTH
                        Generate image width (default: 512)
  --image_height IMAGE_HEIGHT
                        Generate image height (default: 512)
  --num_datasets NUM_DATASETS
                        Number of datasets to generate (default: 1)
  --image_batch_size IMAGE_BATCH_SIZE
                        Number of batches (default: 1)
  --prompt_list_dataset_path PROMPT_LIST_DATASET_PATH
                        The path to prompt list dataset zip
```

``` shell
python3 ./scripts/generate_images_from_prompt_list.py --checkpoint_path "./input/model/sd/v1-5-pruned-emaonly/v1-5-pruned-emaonly.safetensors" --cfg_scale 7 --num_images 2 --output ./output/generated-dataset-from-prompt-list --prompt_list_dataset_path ./test/test_zip_files/prompt_list_civitai_2_test.zip 
```

## Preparing Image Dataset
#### Validate Image Dataset
```
usage: process_dataset.py dataset-validate [-h] [--image-dataset-path IMAGE_DATASET_PATH] [--is-tagged IS_TAGGED] [--is-generated-dataset IS_GENERATED_DATASET]

CLI tool for validating image dataset storage format

options:
  -h, --help            show this help message and exit
  --image-dataset-path IMAGE_DATASET_PATH
                        The path of the image dataset to validate
  --is-tagged IS_TAGGED
                        True if dataset is a tagged dataset (default=False)
  --is-generated-dataset IS_GENERATED_DATASET
                        True if dataset is a generated dataset using kcg-ml-sd1p4's generate images using prompt generator (default=False)

```

Sample command for untagged dataset: 

    python ./scripts/image_dataset_storage_format/process_dataset.py dataset-validate --image-dataset-path "./test/test_zip_files/test-dataset-correct-format.zip"

Sample command for tagged dataset: 

    python ./scripts/image_dataset_storage_format/process_dataset.py dataset-validate --image-dataset-path "./test/test_zip_files/test-dataset-correct-format-tagged.zip" --is-tagged True

Sample command for validating generated dataset from kcg-ml-sd1p4: 

    python ./scripts/image_dataset_storage_format/process_dataset.py dataset-validate --image-dataset-path "./test/test_zip_files/test-generated-dataset-correct-format.zip" --is-generated-dataset True


#### Format and Compute Manifest Of Image Dataset
```
usage: process_dataset.py format-and-compute-manifest [-h] [--image-dataset-path IMAGE_DATASET_PATH] [--is-tagged IS_TAGGED] [--is-generated-dataset IS_GENERATED_DATASET] [--output-path OUTPUT_PATH]

CLI tool for formatting images and computing manifest of dataset

options:
  -h, --help            show this help message and exit
  --image-dataset-path IMAGE_DATASET_PATH
                        The path of the image dataset to process
  --is-tagged IS_TAGGED
                        True if dataset is a tagged dataset (default=False)
  --is-generated-dataset IS_GENERATED_DATASET
                        True if dataset is a generated dataset using kcg-ml-sd1p4's generate images using prompt generator (default=False)
  --output-path OUTPUT_PATH
                        The path to save the dataset (default='./output')
```

Sample command for untagged dataset: 

    python ./scripts/image_dataset_storage_format/process_dataset.py format-and-compute-manifest --image-dataset-path "./test/test_zip_files/test-dataset-not-all-images-are-in-images-dir.zip" --output-path="./output"

Sample command for tagged dataset: 

    python ./scripts/image_dataset_storage_format/process_dataset.py format-and-compute-manifest --image-dataset-path "./test/test_zip_files/test-dataset-no-features-manifest-tagged.zip" --is-tagged True --output-path="./output"

Sample command for validating generated dataset from kcg-ml-sd1p4: 

    python ./scripts/image_dataset_storage_format/process_dataset.py format-and-compute-manifest --image-dataset-path "./test/test_zip_files/test-generated-dataset-correct-format.zip" --output-path="./output" --is-generated-dataset True

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
