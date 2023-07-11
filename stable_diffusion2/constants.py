import os
import sys

base_directory = "./"
sys.path.insert(0, base_directory)

ROOT_MODELS_PATH = os.path.abspath(os.path.join(base_directory, 'input/model/'))

CHECKPOINT_PATH = os.path.abspath(os.path.join(ROOT_MODELS_PATH, 'v1-5-pruned-emaonly.ckpt'))

EMBEDDER_PATH = os.path.abspath(os.path.join(ROOT_MODELS_PATH, 'clip/clip_embedder.ckpt'))
TOKENIZER_PATH = os.path.abspath(os.path.join(ROOT_MODELS_PATH, 'clip/clip_tokenizer.ckpt'))
TRANSFORMER_PATH = os.path.abspath(os.path.join(ROOT_MODELS_PATH, 'clip/clip_transformer.ckpt'))

UNET_PATH = os.path.abspath(os.path.join(ROOT_MODELS_PATH, 'unet/unet.ckpt'))

AUTOENCODER_PATH = os.path.abspath(os.path.join(ROOT_MODELS_PATH, 'autoencoder/autoencoder.ckpt'))
ENCODER_PATH = os.path.abspath(os.path.join(ROOT_MODELS_PATH, 'autoencoder/encoder.ckpt'))
DECODER_PATH = os.path.abspath(os.path.join(ROOT_MODELS_PATH, 'autoencoder/decoder.ckpt'))

LATENT_DIFFUSION_PATH = os.path.abspath(os.path.join(ROOT_MODELS_PATH, 'latent_diffusion/latent_diffusion.ckpt'))