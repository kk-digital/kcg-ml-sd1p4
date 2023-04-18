# kcg-ml-sd1p4

### Notebooks
| Notebook Title | Google Colab Link |
| --- | --- |
| Diffusers Unit Test Example | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kk-digital/kcg-ml-sd1p4/blob/main/notebooks/diffusers_unit_test.ipynb)|
## Cleaning Jupyter Notebooks for Version Control
### Installation
```sh
pip install nbstripout
```
#### Add a git filter to your repository's .git/confige file. nbsstipout will automatically strip the output and metadata from any notebook files
```sh
nbstripout --install
```

## Downloading Models

##### v1-5-pruned-emaonly.safetensors
``` bash 
!wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors 
```

##### sd-v1-4.ckpt
```
!wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt 
``` 



