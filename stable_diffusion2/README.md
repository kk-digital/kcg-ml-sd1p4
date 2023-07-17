# kcg-ml-sd1p4

[![Build Status](http://103.68.223.207:8111/app/rest/builds/buildType:(id:KcgMlSd1p4_Build)/statusIcon)](http://teamcity/viewType.html?buildTypeId=KcgMlSd1p4_Build&guest=1)

## Summary

- [kcg-ml-sd1p4](#kcg-ml-sd1p4)
  - [Summary](#summary)
  - [Downloading models](#downloading-models)
  - [Saving submodels](#saving-submodels)
  - [Running `stable_diffusion2` scripts](#running-stable_diffusion2-scripts)
      - [Embed prompts](#embed-prompts)
      - [Images from embeddings](#images-from-embeddings)
      - [Images from distributions](#images-from-distributions)
      - [Images from temperature range](#images-from-temperature-range)
      - [Images and encodings](#images-and-encodings)
  - [Notebooks](#notebooks)
## Downloading models

Will download the currently supported checkpoint, `v1-5-pruned-emaonly.ckpt`, to `./input/model/`.

```bash
./stable_diffusion2/download-model.sh
```
## Saving submodels

This initializes a `LatentDiffusion` module, load its weights from a checkpoint file at `./input/model/v1-5-pruned-emaonly.ckpt` and then save all submodules.
**Note**: saving these models will take an extra ~5GB of storage. 

```bash
python3 ./scripts2/save_models.py
```

**Command line arguments**
- `-g, --granularity`: Determines the height in the model tree on which to save the submodels. Defaults to `0`, saving all submodels of the `LatentDiffusion` class and all submodels thereof. If `1` is given, it saves only up to the first level, i.e., autoencoder, UNet and CLIP text embedder. _Note that_ the other scripts expect this to be ran with `0`.
- `--root_models_path`: Base directory for the models' folder structure. Defaults to the constant `ROOT_MODELS_PATH`, which is expected to be `./input/model/`.
- `--checkpoint_path`: Relative path of the checkpoint file (*.ckpt). Defaults to the constant `CHECKPOINT_PATH`, which is expected to be `'./input/model/v1-5-pruned-emaonly.ckpt' = os.path.join(ROOT_MODELS_PATH, 'v1-5-pruned-emaonly.ckpt')`.

## Running `stable_diffusion2` scripts

There are five other new scripts besides `scripts2/save_models.py`.

#### Embed prompts
Command:

```bash
python3 ./scripts2/embed_prompts.py --prompts 'A painting of a computer virus', 'An old photo of a computer scientist', 'A computer drawing a computer'
```

Saves a tensor of a batch of prompts embeddings, and the tensor for the null prompt `""`.
- `-p, --prompts`: The prompts to embed. Defaults to `['A painting of a computer virus', 'An old photo of a computer scientist']`.
- `--embedded_prompts_dir`: The path to the directory containing the embedded prompts tensors. Defaults to a constant `EMBEDDED_PROMPTS_DIR`, which is expected to be `'./input/embedded_prompts/'`.
#### Images from embeddings
Only run _after_ generating the embedded prompts with the above script.
Try running:
```bash
python3 ./scripts2/generate_images_from_embeddings.py --num_seeds 4 --temperature 1.2 --ddim_eta 0.2
```

**Command line arguments**

- `-p, --embedded_prompts_dir`: The path to the directory containing the embedded prompts tensors. Defaults to the `EMBEDDED_PROMPTS_DIR` constant, which is expected to be `'./input/embedded_prompts/'`.
- `-od, --output_dir`: The output directory. Defaults to the `OUTPUT_DIR` constant, which is expected to be `'./output/noise-tests/from_embeddings'`.
- `--num_seeds`: Number of random seeds to use. Defaults to `3`. Ranges from `1` to `7`.
- `-bs, --batch_size`: Batch size to use. Defaults to `1`. 
- `-t, --temperature`: Sampling temperature. Defaults to `1.0`.
- `--ddim_eta`: Amount of noise to readd during the sampling process. Defaults to `0.0`.
- `--clear_output_dir`: Either to clear or not the output directory before running. Defaults to `False`.
- `--cuda_device`: CUDA device to use. Defaults to `cuda:0`.
 
#### Images from distributions
Try running:
```bash
python3 ./scripts2/generate_images_from_temperature_range.py -d 4 --params_steps 4 --params_range 0.49, 0.54 --num_seeds 4 --temperature 1.2 --ddim_eta 1.2
```
#### Images from temperature range
Try running:
```bash
python3 ./scripts2/generate_images_from_temperature_range.py -d 4 --params_range 0.49 0.54 --params_steps 3 --temperature_steps 3 --temperature_range 0.8 2.0
```
#### Images and encodings
Try running:
```bash
python3 ./scripts2/generate_images_and_encodings.py --prompt "A computer virus painting a minimalist paintwork" --num_iterations 10
```
## Notebooks
