#### naming conventions
Whenever it seemed useful to keep the original files as reference, I've copied them and renamed the copies to `*2.py` or `*2/` (`.py` files and folders, respectively). For instance, the refactored package is contained in a folder named `stable_diffusion2/`; there's a folder named `model2/`; there's a Python file named `model2.py`. *This is temporary*. More than just a reference, keeping the original files allows us to more easily compare the results we get from the two pipelines during the development phase.

#### design choices
I aimed to find a common ground between the previous codebase and the new pipeline. So I kept everything that subclassed a `nn.Module` as a subclass of a `nn.Module`. That implies that, for those classes, a `load`/`unload` function, as a class instance method, is in general pointless. For that we already have a `load_state_dict` function. Instead, I added `load_submodels`, `unload_submodels` and `save_submodels` functions, when applicable, and a `save` function, to save the whole `nn.Module` to disk (*there will be examples*).
Classes for models that support submodels have their submodels as kwargs (defaulting to `None`) for their `__init__` function. You can either pass the externally initialized/loaded submodels at class instantiation or instantiate the class without providing the submodels and then load them from disk with `load_submodels`.


#### model hierarchy

The classes I consider to be models, and their corresponding submodels are:
- `stable_diffusion2.latent_diffusion.LatentDiffusion` 
  - `stable_diffusion2.model.unet.UNetModel`
  - `stable_diffusion2.model.vae.Autoencoder`
    - `stable_diffusion2.model.vae.Encoder`
    - `stable_diffusion2.model.vae.Decoder`
  - `stable_diffusion2.model.clip.CLIPTextEmbedder`
    - `transformers.CLIPTokenizer`
    - `transformers.CLIPTextModel`

Classes with submodels have `load_submodels` and `save_submodels` functions.


#### folder structure
The current directory tree for the `stable_diffusion2` folder is as follows:

```bash
├── __init__.py
├── constants.py
├── latent_diffusion.py
├── model
│   ├── __init__.py
│   ├── autoencoder.py
│   ├── clip_embedder.py
│   ├── unet.py
│   └── unet_attention.py
├── model2
│   ├── __init__.py
│   ├── clip
│   │   ├── __init__.py
│   │   └── clip_embedder.py
│   ├── unet
│   │   ├── __init__.py
│   │   ├── unet.py
│   │   └── unet_attention.py
│   └── vae
│       ├── __init__.py
│       ├── autoencoder.py
│       ├── auxiliary_classes.py
│       ├── decoder.py
│       └── encoder.py
├── sampler
│   ├── __init__.py
│   ├── ddim.py
│   └── ddpm.py
└── utils
    ├── __init__.py
    ├── cli.py
    ├── model.py
    ├── model2.py
    └── utils.py
```
The folder structure of `model/` was refined (compare to `model2/`). Now there's one folder per model, which contains files where the model (and its corresponding submodels, when applicable) classes are defined.

#### non-exhaustive overview of the already implemented changes

##### @`stable_diffusion2/` 
- Added `constants.py`: it has the default paths for saving/loading models.
- Changed `latent_diffusion.py`: `LatentDiffusion` class now has `save`, `save_submodels`, `load_submodels` and `unload_submodels` methods.

##### @`stable_diffusion2/model2/vae/`
- Changed `autoencoder.py`: `Autoencoder` class now has `save`, `save_submodels`, `initialize_submodels`, `load_submodels` and `unload_submodels` methods; moved `Encoder` and `Decoder` classes to separated files (`decoder.py` and `encoder.py`); moved other classes to `auxiliary_classes.py`.
- `Encoder` and `Decoder` classes now have a `save` method.

##### @`stable_diffusion2/model2/clip/`
- Changed `clip_embedder.py`: `CLIPTextEmbedder` class now has `save`, `save_submodels`, `load_submodels`, `unload_submodels`, `load_tokenizer_from_lib`, `load_transformer_from_lib` methods.

##### @`stable_diffusion2/model2/unet/`
- Changed `unet.py`: `UNetModel` class now has a `save` function.

##### @`stable_diffusion2/sampler/`
- For both `ddim.py` and `ddpm.py`: `sample` and `p_sample` methods now take an extra kwarg `noise_fn`, the probability distribution from which to sample noise vectors during the diffusion process; for `ddim.py` that also applies to the `get_x_prev_and_pred_x0` method.

##### @`stable_diffusion2/utils/`
- Changed `model2.py`:
- Added `utils.py`: