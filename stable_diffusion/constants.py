import os
import sys
import dataclasses

# base_directory = "./"
# sys.path.insert(0, base_directory)

base_directory = os.getcwd()
sys.path.insert(0, base_directory)

ROOT_MODELS_PREFIX = "./input/model/"
CHECKPOINT = "v1-5-pruned-emaonly.ckpt"

ROOT_MODELS_PATH = os.path.abspath(os.path.join(base_directory, ROOT_MODELS_PREFIX))

CHECKPOINT_PATH = os.path.abspath(
    os.path.join(ROOT_MODELS_PATH, CHECKPOINT)
)

EMBEDDER_PATH = os.path.abspath(
    os.path.join(ROOT_MODELS_PATH, "clip_text_embedder/clip_embedder.ckpt")
)
TOKENIZER_PATH = os.path.abspath(
    os.path.join(ROOT_MODELS_PATH, "clip_text_embedder/clip_tokenizer.ckpt")
)
TRANSFORMER_PATH = os.path.abspath(
    os.path.join(ROOT_MODELS_PATH, "clip_text_embedder/clip_transformer.ckpt")
)

IMAGE_PROCESSOR_PATH = os.path.abspath(
    os.path.join(ROOT_MODELS_PATH, "clip_image_encoder/image_processor.ckpt")
)
CLIP_MODEL_PATH = os.path.abspath(
    os.path.join(ROOT_MODELS_PATH, "clip_image_encoder/clip_model.ckpt")
)
IMAGE_ENCODER_PATH = os.path.abspath(
    os.path.join(ROOT_MODELS_PATH, "clip_image_encoder/clip_image_encoder.ckpt")
)

UNET_PATH = os.path.abspath(os.path.join(ROOT_MODELS_PATH, "unet/unet.ckpt"))

AUTOENCODER_PATH = os.path.abspath(
    os.path.join(ROOT_MODELS_PATH, "autoencoder/autoencoder.ckpt")
)
ENCODER_PATH = os.path.abspath(
    os.path.join(ROOT_MODELS_PATH, "autoencoder/encoder.ckpt")
)
DECODER_PATH = os.path.abspath(
    os.path.join(ROOT_MODELS_PATH, "autoencoder/decoder.ckpt")
)

LATENT_DIFFUSION_PATH = os.path.abspath(
    os.path.join(ROOT_MODELS_PATH, "latent_diffusion/latent_diffusion.ckpt")
)


class ModelsPathTree:
    """returns dicts to be used as kwargs for loading submodels.
    the keys are the same as the kwargs used for the load methods."""

    def __init__(self, base_directory: str = "./"):
        self.base_directory = base_directory
        self.root_models_path = os.path.abspath(
            os.path.join(base_directory, ROOT_MODELS_PREFIX)
        )
        self.checkpoint_path = os.path.abspath(
            os.path.join(self.root_models_path, CHECKPOINT)
        )
        self.embedder_path = os.path.abspath(
            os.path.join(self.root_models_path, "clip_text_embedder/clip_embedder.ckpt")
        )
        self.tokenizer_path = os.path.abspath(
            os.path.join(
                self.root_models_path, "clip_text_embedder/clip_tokenizer.ckpt"
            )
        )
        self.transformer_path = os.path.abspath(
            os.path.join(
                self.root_models_path, "clip_text_embedder/clip_transformer.ckpt"
            )
        )
        self.image_encoder_path = os.path.abspath(
            os.path.join(
                self.root_models_path, "clip_image_encoder/clip_image_encoder.ckpt"
            )
        )
        self.clip_model_path = os.path.abspath(
            os.path.join(self.root_models_path, "clip_image_encoder/clip_model.ckpt")
        )
        self.image_processor_path = os.path.abspath(
            os.path.join(
                self.root_models_path, "clip_image_encoder/image_processor.ckpt"
            )
        )
        self.unet_path = os.path.abspath(
            os.path.join(self.root_models_path, "unet/unet.ckpt")
        )
        self.autoencoder_path = os.path.abspath(
            os.path.join(self.root_models_path, "autoencoder/autoencoder.ckpt")
        )
        self.encoder_path = os.path.abspath(
            os.path.join(self.root_models_path, "autoencoder/encoder.ckpt")
        )
        self.decoder_path = os.path.abspath(
            os.path.join(self.root_models_path, "autoencoder/decoder.ckpt")
        )
        self.latent_diffusion_path = os.path.abspath(
            os.path.join(
                self.root_models_path, "latent_diffusion/latent_diffusion.ckpt"
            )
        )

    def __repr__(self):
        return f"""ModelsPathTree(base_directory={self.base_directory})
                - latent_diffusion: {self.latent_diffusion_path}
                
                -- autoencoder: {self.autoencoder_path}
                --- encoder: {self.encoder_path}
                --- decoder: {self.decoder_path}
                
                -- embedder: {self.embedder_path}
                --- tokenizer: {self.tokenizer_path}
                --- transformer: {self.transformer_path}
                
                -- unet: {self.unet_path}

                - image_encoder: {self.image_encoder_path}
                -- clip_model: {self.clip_model_path}
                -- image_processor: {self.image_processor_path}
            """

    @property
    def encoder(self):
        """ "returns a dict {encoder_path: self.encoder_path}"""
        return {"encoder_path": self.encoder_path}

    @property
    def decoder(self):
        """returns a dict {decoder_path: self.decoder_path}"""
        return {"decoder_path": self.decoder_path}

    @property
    def autoencoder(self):
        """returns a dict {autoencoder_path: self.autoencoder_path}"""
        return {"autoencoder_path": self.autoencoder_path}

    @property
    def unet(self):
        """returns a dict {unet_path: self.unet_path}"""
        return {"unet_path": self.unet_path}

    @property
    def embedder(self):
        """returns a dict {embedder_path: self.embedder_path}"""
        return {"embedder_path": self.embedder_path}

    @property
    def tokenizer(self):
        """returns a dict {tokenizer_path: self.tokenizer_path}"""
        return {"tokenizer_path": self.tokenizer_path}

    @property
    def transformer(self):
        """returns a dict {transformer_path: self.transformer_path}"""
        return {"transformer_path": self.transformer_path}

    @property
    def image_processor(self):
        """returns a dict {image_processor_path: self.image_processor_path}"""
        return {"image_processor_path": self.image_processor_path}

    @property
    def clip_model(self):
        """returns a dict {clip_model_path: self.clip_model_path}"""
        return {"clip_model_path": self.clip_model_path}

    @property
    def image_encoder(self):
        return {"image_encoder_path": self.image_encoder_path}

    @property
    def latent_diffusion(self):
        return {"latent_diffusion_path": self.latent_diffusion_path}

    @property
    def latent_diffusion_submodels(self):
        return {
            "embedder_path": self.embedder_path,
            "unet_path": self.unet_path,
            "autoencoder_path": self.autoencoder_path,
        }

    @property
    def latent_diffusion_submodels_tree(self):
        return {
            "embedder_path": self.embedder_path,
            "tokenizer_path": self.tokenizer_path,
            "transformer_path": self.transformer_path,
            "unet_path": self.unet_path,
            "autoencoder_path": self.autoencoder_path,
            "encoder_path": self.encoder_path,
            "decoder_path": self.decoder_path,
        }

    @property
    def embedder_submodels(self):
        return {
            "tokenizer_path": self.tokenizer_path,
            "transformer_path": self.transformer_path,
        }

    @property
    def image_encoder_submodels(self):
        return {
            "clip_model_path": self.clip_model_path,
            "image_processor_path": self.image_processor_path,
        }

    @property
    def autoencoder_submodels(self):
        return {"encoder_path": self.encoder_path, "decoder_path": self.decoder_path}


def create_directory_tree_paths(base_directory: str = "./"):
    ROOT_MODELS_PATH = os.path.abspath(os.path.join(base_directory, ROOT_MODELS_PREFIX))

    CHECKPOINT_PATH = os.path.abspath(
        os.path.join(ROOT_MODELS_PATH, CHECKPOINT)
    )

    EMBEDDER_PATH = os.path.abspath(
        os.path.join(ROOT_MODELS_PATH, "clip_text_embedder/clip_embedder.ckpt")
    )
    TOKENIZER_PATH = os.path.abspath(
        os.path.join(ROOT_MODELS_PATH, "clip_text_embedder/clip_tokenizer.ckpt")
    )
    TRANSFORMER_PATH = os.path.abspath(
        os.path.join(ROOT_MODELS_PATH, "clip_text_embedder/clip_transformer.ckpt")
    )

    IMAGE_PROCESSOR_PATH = os.path.abspath(
        os.path.join(ROOT_MODELS_PATH, "clip_image_encoder/image_processor.ckpt")
    )
    CLIP_MODEL_PATH = os.path.abspath(
        os.path.join(ROOT_MODELS_PATH, "clip_image_encoder/clip_model.ckpt")
    )
    IMAGE_ENCODER_PATH = os.path.abspath(
        os.path.join(ROOT_MODELS_PATH, "clip_image_encoder/clip_image_encoder.ckpt")
    )

    UNET_PATH = os.path.abspath(os.path.join(ROOT_MODELS_PATH, "unet/unet.ckpt"))

    AUTOENCODER_PATH = os.path.abspath(
        os.path.join(ROOT_MODELS_PATH, "autoencoder/autoencoder.ckpt")
    )
    ENCODER_PATH = os.path.abspath(
        os.path.join(ROOT_MODELS_PATH, "autoencoder/encoder.ckpt")
    )
    DECODER_PATH = os.path.abspath(
        os.path.join(ROOT_MODELS_PATH, "autoencoder/decoder.ckpt")
    )

    LATENT_DIFFUSION_PATH = os.path.abspath(
        os.path.join(ROOT_MODELS_PATH, "latent_diffusion/latent_diffusion.ckpt")
    )

    paths_dict = {
        "root_models_path": ROOT_MODELS_PATH,
        "checkpoint_path": CHECKPOINT_PATH,
        "embedder_path": EMBEDDER_PATH,
        "embedder_submodels_paths": {
            "tokenizer_path": TOKENIZER_PATH,
            "transformer_path": TRANSFORMER_PATH,
        },
        "image_encoder_path": IMAGE_ENCODER_PATH,
        "image_encoder_submodels_paths": {
            "image_processor_path": IMAGE_PROCESSOR_PATH,
            "clip_model_path": CLIP_MODEL_PATH,
        },
        "unet_path": UNET_PATH,
        "autoencoder_path": AUTOENCODER_PATH,
        "autoencoder_submodels_paths": {
            "encoder_path": ENCODER_PATH,
            "decoder_path": DECODER_PATH,
        },
        "latent_diffusion_path": LATENT_DIFFUSION_PATH,
        "latent_diffusion_submodels_paths": {
            "embedder_path": EMBEDDER_PATH,
            "unet_path": UNET_PATH,
            "autoencoder_path": AUTOENCODER_PATH,
        },
    }

    return paths_dict


def create_latent_diffusion_submodels_tree_paths(base_directory: str = "./"):
    ROOT_MODELS_PATH = os.path.abspath(os.path.join(base_directory, ROOT_MODELS_PREFIX))

    EMBEDDER_PATH = os.path.abspath(
        os.path.join(ROOT_MODELS_PATH, "clip_text_embedder/clip_embedder.ckpt")
    )
    TOKENIZER_PATH = os.path.abspath(
        os.path.join(ROOT_MODELS_PATH, "clip_text_embedder/clip_tokenizer.ckpt")
    )
    TRANSFORMER_PATH = os.path.abspath(
        os.path.join(ROOT_MODELS_PATH, "clip_text_embedder/clip_transformer.ckpt")
    )

    UNET_PATH = os.path.abspath(os.path.join(ROOT_MODELS_PATH, "unet/unet.ckpt"))

    AUTOENCODER_PATH = os.path.abspath(
        os.path.join(ROOT_MODELS_PATH, "autoencoder/autoencoder.ckpt")
    )
    ENCODER_PATH = os.path.abspath(
        os.path.join(ROOT_MODELS_PATH, "autoencoder/encoder.ckpt")
    )
    DECODER_PATH = os.path.abspath(
        os.path.join(ROOT_MODELS_PATH, "autoencoder/decoder.ckpt")
    )

    paths_dict = {
        "embedder_path": EMBEDDER_PATH,
        "tokenizer_path": TOKENIZER_PATH,
        "transformer_path": TRANSFORMER_PATH,
        "autoencoder_path": AUTOENCODER_PATH,
        "encoder_path": ENCODER_PATH,
        "decoder_path": DECODER_PATH,
        "unet_path": UNET_PATH,
    }

    return paths_dict
