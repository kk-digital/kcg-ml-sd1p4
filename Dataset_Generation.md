# Dataset Generation

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
