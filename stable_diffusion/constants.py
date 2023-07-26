import os
import sys
import dataclasses

# base_directory = "./"
# sys.path.insert(0, base_directory)

base_directory = os.getcwd()
sys.path.insert(0, base_directory)
BASE_IO_DIRECTORY = "E:\\ML\\"
ROOT_MODELS_PREFIX = "input/models/"
MODEL = "v1-5-pruned-emaonly"
CHECKPOINT = f"{MODEL}.safetensors"

ROOT_MODELS_PATH = (os.path.join(BASE_IO_DIRECTORY, ROOT_MODELS_PREFIX))
SD_MODELS_DIR = os.path.join(ROOT_MODELS_PATH, "sd")

SD_DEFAULT_MODEL_DIR = os.path.join(SD_MODELS_DIR, MODEL)

CLIP_MODELS_DIR = os.path.join(ROOT_MODELS_PATH, "clip")

CHECKPOINT_PATH = (
    os.path.join(SD_DEFAULT_MODEL_DIR, CHECKPOINT)
)

EMBEDDER_PATH = (
    os.path.join(CLIP_MODELS_DIR, "clip_text_embedder/clip_embedder.ckpt")
)
TOKENIZER_PATH = (
    os.path.join(CLIP_MODELS_DIR, "clip_text_embedder/clip_tokenizer.ckpt")
)
TRANSFORMER_PATH = (
    os.path.join(CLIP_MODELS_DIR, "clip_text_embedder/clip_transformer.ckpt")
)

IMAGE_PROCESSOR_PATH = (
    os.path.join(CLIP_MODELS_DIR, "clip_image_encoder/image_processor.ckpt")
)
CLIP_MODEL_PATH = (
    os.path.join(CLIP_MODELS_DIR, "clip_image_encoder/clip_model.ckpt")
)
IMAGE_ENCODER_PATH = (
    os.path.join(CLIP_MODELS_DIR, "clip_image_encoder/clip_image_encoder.ckpt")
)

TEXT_EMBEDDER_PATH = (
    os.path.join(SD_DEFAULT_MODEL_DIR, "clip_text_embedder.ckpt")
)

UNET_PATH = (os.path.join(SD_DEFAULT_MODEL_DIR, "unet.safetensors"))

AUTOENCODER_PATH = (
    os.path.join(SD_DEFAULT_MODEL_DIR, "autoencoder.safetensors")
)
ENCODER_PATH = (
    os.path.join(SD_DEFAULT_MODEL_DIR, "encoder.safetensors")
)
DECODER_PATH = (
    os.path.join(SD_DEFAULT_MODEL_DIR, "decoder.safetensors")
)
LATENT_DIFFUSION_PATH = (
    os.path.join(SD_DEFAULT_MODEL_DIR, "latent_diffusion.safetensors")
)


class ModelsPathTree:
    """returns dicts to be used as kwargs for loading submodels.
    the keys are the same as the kwargs used for the load methods."""

    def __init__(self, base_io_directory: str = "./", base_directory: str = "./", model_name = "v1-5-pruned-emaonly", root_models_prefix: str = "input/models/", root_outputs_prefix: str = "outputs/models/", checkpoint_format = ".safetensors"):
        
        self.base_directory = base_directory
        self.base_io_directory = base_io_directory
        self.checkpoint_format = checkpoint_format
        self.root_models_prefix = root_models_prefix
        self.model_name = model_name
        self.model_checkpoint = f"{self.model_name}{self.checkpoint_format}"
        self.root_models_dir = (
            os.path.join(base_io_directory, root_models_prefix)
        )
        self.root_sd_models_dir = (
            os.path.join(self.root_models_dir, "sd")
        )
        self.root_sd_model_dir = (
            os.path.join(self.root_models_dir, self.model_name)
        )        
        self.root_clip_models_dir = (
            os.path.join(self.root_models_dir, "clip")
        )
        self.root_outputs_prefix = root_outputs_prefix
        self.root_outputs_dir = (
            os.path.join(base_io_directory, root_outputs_prefix)
            )
        self.root_model_outputs_dir = (
            os.path.join(self.root_outputs_dir, self.model_name)
        )
        self.checkpoint_path = (
            os.path.join(self.root_sd_model_dir, self.model_checkpoint)
        )    
        self.embedder_path = (
            os.path.join(self.root_sd_model_dir, "clip_text_embedder.ckpt")
        )
        self.tokenizer_path = (
            os.path.join(
                self.root_models_dir, "clip_text_embedder/tokenizer"
            )
        )
        self.transformer_path = (
            os.path.join(
                self.root_models_dir, "clip_text_embedder/clip_transformer.ckpt"
            )
        )
        self.image_encoder_path = (
            os.path.join(
                self.root_models_dir, "clip_image_encoder/clip_image_encoder.ckpt"
            )
        )
        self.clip_model_path = (
            os.path.join(self.root_models_dir, "clip_image_encoder/clip_model.ckpt")
        )
        self.image_processor_path = (
            os.path.join(
                self.root_models_dir, "clip_image_encoder/image_processor.ckpt"
            )
        )
        self.unet_path = (
            os.path.join(self.root_sd_model_dir, "unet.safetensors")
        )
        self.autoencoder_path = (
            os.path.join(self.root_sd_model_dir, "autoencoder.safetensors")
        )
        self.encoder_path = (
            os.path.join(self.root_sd_model_dir, "encoder.safetensors")
        )
        self.decoder_path = (
            os.path.join(self.root_sd_model_dir, "decoder.safetensors")
        )
        self.latent_diffusion_path = (
            os.path.join(
                self.root_sd_model_dir, "latent_diffusion.safetensors"
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
    def checkpoint(self):
        """ "returns a dict {checkpoint_path: self.checkpoint_path}"""
        return {"checkpoint_path": self.checkpoint_path}

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
    ROOT_MODELS_PATH = (os.path.join(base_directory, ROOT_MODELS_PREFIX))

    CHECKPOINT_PATH = (
        os.path.join(ROOT_MODELS_PATH, CHECKPOINT)
    )
    EMBEDDER_PATH = (
        os.path.join(ROOT_MODELS_PATH, "clip_text_embedder/clip_embedder.ckpt")
    )
    TOKENIZER_PATH = (
        os.path.join(ROOT_MODELS_PATH, "clip_text_embedder/clip_tokenizer.ckpt")
    )
    TRANSFORMER_PATH = (
        os.path.join(ROOT_MODELS_PATH, "clip_text_embedder/clip_transformer.ckpt")
    )

    IMAGE_PROCESSOR_PATH = (
        os.path.join(ROOT_MODELS_PATH, "clip_image_encoder/image_processor.ckpt")
    )
    CLIP_MODEL_PATH = (
        os.path.join(ROOT_MODELS_PATH, "clip_image_encoder/clip_model.ckpt")
    )
    IMAGE_ENCODER_PATH = (
        os.path.join(ROOT_MODELS_PATH, "clip_image_encoder/clip_image_encoder.ckpt")
    )

    UNET_PATH = (os.path.join(ROOT_MODELS_PATH, "unet/unet.safetensors"))

    AUTOENCODER_PATH = (
        os.path.join(ROOT_MODELS_PATH, "autoencoder/autoencoder.safetensors")
    )
    ENCODER_PATH = (
        os.path.join(ROOT_MODELS_PATH, "autoencoder/encoder.safetensors")
    )
    DECODER_PATH = (
        os.path.join(ROOT_MODELS_PATH, "autoencoder/decoder.safetensors")
    )

    LATENT_DIFFUSION_PATH = (
        os.path.join(ROOT_MODELS_PATH, "latent_diffusion/latent_diffusion.safetensors")
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
    ROOT_MODELS_PATH = (os.path.join(base_directory, ROOT_MODELS_PREFIX))

    EMBEDDER_PATH = (
        os.path.join(ROOT_MODELS_PATH, "clip_text_embedder/clip_embedder.ckpt")
    )
    TOKENIZER_PATH = (
        os.path.join(ROOT_MODELS_PATH, "clip_text_embedder/clip_tokenizer.ckpt")
    )
    TRANSFORMER_PATH = (
        os.path.join(ROOT_MODELS_PATH, "clip_text_embedder/clip_transformer.ckpt")
    )

    UNET_PATH = (os.path.join(ROOT_MODELS_PATH, "unet/unet.safetensors"))

    AUTOENCODER_PATH = (
        os.path.join(ROOT_MODELS_PATH, "autoencoder/autoencoder.safetensors")
    )
    ENCODER_PATH = (
        os.path.join(ROOT_MODELS_PATH, "autoencoder/encoder.safetensors")
    )
    DECODER_PATH = (
        os.path.join(ROOT_MODELS_PATH, "autoencoder/decoder.safetensors")
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
