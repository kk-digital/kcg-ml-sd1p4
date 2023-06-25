import os

CHECKPOINT_PATH = os.path.abspath('./input/model/v1-5-pruned-emaonly.ckpt')

EMBEDDER_PATH = os.path.abspath('./input/model/clip/clip_embedder.ckpt')
TOKENIZER_PATH = os.path.abspath('./input/model/clip/clip_tokenizer.ckpt')
TRANSFORMER_PATH = os.path.abspath('./input/model/clip/clip_transformer.ckpt')

UNET_PATH = os.path.abspath('./input/model/unet/unet.ckpt')

AUTOENCODER_PATH = os.path.abspath('./input/model/autoencoder/autoencoder.ckpt')
ENCODER_PATH = os.path.abspath('./input/model/autoencoder/encoder.ckpt')
DECODER_PATH = os.path.abspath('./input/model/autoencoder/decoder.ckpt')