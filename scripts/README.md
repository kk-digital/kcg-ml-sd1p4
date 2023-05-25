# DataUtility folder

## What is this?

This folder contains the scripts that are used to generate, download and manage the data used in the project.

## Summary

- [DataUtility folder](#datautility-folder)
  - [What is this?](#what-is-this)
  - [Summary](#summary)
  - [How to use the scripts?](#how-to-use-the-scripts)
    - [download\_model.py](#download_modelpy)

## How to use the scripts?

### download_model.py

**Command line arguments:**

- `--list-models`: List all the available models to download, when this argument is used, the script will ignore all other arguments and only list the models.
- `--model`: Name of the model to download, this argument is required when `--list-models` is not used.
- `--output`: Destination to save the model to, this argument is optional and defaults to `/tmp/input/models/`.

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
