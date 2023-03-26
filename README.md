# kcg-ml-sd1p4

### Notebooks
| Notebook Title | Google Colab Link |
| --- | --- |
| Diffusers Unit Test Example | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kk-digital/kcg-ml-sd1p4/blob/main/notebooks/diffusers_unit_test.ipynb)|
## Cleaning Jupyter Notebooks for Version Control
When working with Jupyter notebooks in a version control system such as Git, it's often useful to remove the output cells from the notebook files before committing them. This keeps the repository clean and reduces the file size.

To clean Jupyter notebooks for version control, we can use the nbstripout and nbconvert tools.

### Installation
First, make sure you have nbstripout and nbconvert installed . You can install them using pip:
```sh
pip install nbstripout nbconvert
```
### Setting up nbstripout
Next, we'll set up nbstripout to automatically clean the notebooks every time we commit them. Run the following command inside repo directory to install the nbstripout Git filter:
```sh
nbstripout --install
```
Alternative installation to git attributes
```sh
nbstripout --install --attributes .gitattributes
```
### Using nbconvert
Finally, you can use nbconvert to clean the output cells from the notebooks manually. Run the following command to clean all .ipynb files in the current directory:
```sh
python -m nbconvert --ClearOutputPreprocessor.enabled=True --to notebook *.ipynb --inplace
```
