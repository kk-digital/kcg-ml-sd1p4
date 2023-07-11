#### running
The CLI isn't developed, but you can test the codes -- they will need some storage, ~8GB --, after switching to this branch (hence having the adequate folder structure) with:


`python save_all_models.py 0`, then `python save_all_models.py 5 True`

and


`python /scripts2/generate_images_from_noise.py 4`


you might want to drag the outputs out of the `OUTPUT_DIR` since it gets cleaned


then


`python /scripts2/generate_images_from_noise.py 4 True 1`


again, you might want to drag the outputs out of the `OUTPUT_DIR`, then


`python /scripts2/generate_images_from_noise.py 4 True 2`


you get the ouputs of three different initalization procedures for the `LatentDiffusion` class.


#### design choices
I aimed to find a common ground between the previous codebase and the new pipeline. So I kept everything that subclassed a `nn.Module` as a subclass of a `nn.Module`. That implies that, for those classes, a `load`/`unload` function, as a class instance method, is in general pointless*. For that we already have a `load_state_dict` function. Instead, I added `load_submodels`, `unload_submodels` and `save_submodels` methods, when applicable, and a `save` function, to save the whole `nn.Module` to disk (*there will be examples*).
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
├── README.md
├── __init__.py
├── constants.py
├── latent_diffusion.py
├── model
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
    └── utils.py
```
The folder structure of `model/` was refined. Now there's one folder per model, which contains files where the model (and its corresponding submodels, when applicable) classes are defined.

#### non-exhaustive overview of the already implemented changes

##### @`stable_diffusion2/` 
- Added `constants.py`: it has the default paths for saving/loading models.
- Changed `latent_diffusion.py`: `LatentDiffusion` class now has `save`, `save_submodels`, `load_submodels` and `unload_submodels` methods.

##### @`stable_diffusion2/model/vae/`
- Changed `autoencoder.py`: `Autoencoder` class now has `save`, `save_submodels`, `initialize_submodels`, `load_submodels` and `unload_submodels` methods; moved `Encoder` and `Decoder` classes to separated files (`decoder.py` and `encoder.py`); moved other classes to `auxiliary_classes.py`.
- `Encoder` and `Decoder` classes now have a `save` method.

##### @`stable_diffusion2/model/clip/`
- Changed `clip_embedder.py`: `CLIPTextEmbedder` class now has `save`, `save_submodels`, `load_submodels`, `unload_submodels`, `load_tokenizer_from_lib`, `load_transformer_from_lib` methods.

##### @`stable_diffusion2/model/unet/`
- Changed `unet.py`: `UNetModel` class now has a `save` function.

##### @`stable_diffusion2/sampler/`
- For both `ddim.py` and `ddpm.py`: `sample` and `p_sample` methods now take an extra kwarg `noise_fn`, the probability distribution from which to sample noise vectors during the diffusion process; for `ddim.py` that also applies to the `get_x_prev_and_pred_x0` method.

##### @`stable_diffusion2/utils/`
- Changed `model.py`:
- Added `utils.py`:
