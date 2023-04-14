# kcg-ml-sd1p4

### Notebooks
| Notebook Title | Google Colab Link |
| --- | --- |
| Diffusers Unit Test Example | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kk-digital/kcg-ml-sd1p4/blob/main/notebooks/diffusers_unit_test.ipynb)|
## Cleaning Jupyter Notebooks for Version Control
### Installation
First, make sure you have nbstripout and nbconvert installed . You can install them using pip:
```sh
pip install nbstripout nbconvert
```
### Setting up nbstripout

```sh
nbstripout --install
```
Alternative installation to git attributes
```sh
nbstripout --install --attributes .gitattributes
```
### Using nbconvert
```sh
python -m nbconvert --ClearOutputPreprocessor.enabled=True --to notebook *.ipynb --inplace
```

### Create the Docker image

```bash 
podman build -t image-generator .
```

### Run the Docker container with the input and output directories mounted from the host

```bash 
podman run -v /path/to/host/tmp/input:/host/tmp/input -v /path/to/host/tmp/output:/host/tmp/output image-generator
```

### Download our finetuned model weights

```bash 
python3 download.py
```
