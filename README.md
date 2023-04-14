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

### How to build dockerfile with podman

```bash 
podman build -t image-generator .
```

