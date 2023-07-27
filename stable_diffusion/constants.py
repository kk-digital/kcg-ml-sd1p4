import os
import sys
import os
import sys
import configparser
import argparse


config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read("./config.ini")

base = config["BASE"]
# base_directory = "./"
# sys.path.insert(0, base_directory)

BASE_DIRECTORY = os.getcwd()

sys.path.insert(0, BASE_DIRECTORY)


BASE_IO_DIRECTORY = base.get("base_io_directory")
ROOT_MODELS_PREFIX = base.get("root_models_prefix")
ROOT_OUTPUTS_PREFIX = base.get("root_outputs_prefix")
MODEL = base.get("model_name")
CLIP_MODEL = base.get("clip_model_name")

CHECKPOINT = f"{MODEL}.safetensors"

ROOT_MODELS_DIR = (os.path.join(BASE_IO_DIRECTORY, ROOT_MODELS_PREFIX))
ROOT_OUTPUTS_DIR = (os.path.join(BASE_IO_DIRECTORY, ROOT_OUTPUTS_PREFIX))
SD_DEFAULT_MODEL_DIR = os.path.join(ROOT_MODELS_DIR, MODEL)
CLIP_MODELS_DIR = os.path.join(ROOT_MODELS_DIR, "clip")
TEXT_EMBEDDER_DIR = (
    os.path.join(CLIP_MODELS_DIR, "text_embedder/")
)
IMAGE_ENCODER_DIR = (
    os.path.join(CLIP_MODELS_DIR, "image_encoder/")
)
CHECKPOINT_PATH = (
    os.path.join(ROOT_MODELS_DIR, CHECKPOINT)
)
TEXT_EMBEDDER_PATH = (
    os.path.join(TEXT_EMBEDDER_DIR, "text_embedder.safetensors")
)
TOKENIZER_PATH = (
    os.path.join(TEXT_EMBEDDER_DIR, "tokenizer/")
)
TEXT_MODEL_PATH = (
    os.path.join(TEXT_EMBEDDER_DIR, CLIP_MODEL)
)

IMAGE_PROCESSOR_PATH = (
    os.path.join(IMAGE_ENCODER_DIR, "image_processor.ckpt")
)
CLIP_MODEL_PATH = (
    os.path.join(IMAGE_ENCODER_DIR, "clip_model.ckpt")
)
IMAGE_ENCODER_PATH = (
    os.path.join(IMAGE_ENCODER_DIR, "clip_image_encoder.ckpt")
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

class IODirectoryTree:
    """returns dicts to be used as kwargs for loading submodels.
    the keys are the same as the kwargs used for the load methods."""

    def __init__(self, 
                base_io_directory: str = BASE_IO_DIRECTORY, 
                base_directory: str = "./", 
                model_name = "v1-5-pruned-emaonly", 
                clip_model_name = "vit-large-patch14", 
                root_models_prefix: str = "input/model/", 
                root_outputs_prefix: str = "output/model/", 
                checkpoint_format = ".safetensors"):
        
        self.base_directory = base_directory
        self.base_io_directory = base_io_directory

        self.checkpoint_format = checkpoint_format

        self.root_models_prefix = root_models_prefix
        self.root_outputs_prefix = root_outputs_prefix

        self.model_name = model_name
        self.model_checkpoint = f"{self.model_name}{self.checkpoint_format}"

        self.clip_model_name = clip_model_name

        self.root_models_dir = (
            os.path.join(base_io_directory, root_models_prefix)
        )
        self.root_outputs_dir = (
            os.path.join(base_io_directory, root_outputs_prefix)
            )
        
        self.sd_model_dir = (
            os.path.join(self.root_models_dir, self.model_name)
        )
        self.clip_models_dir = (
            os.path.join(self.root_models_dir, "clip/")
        )
        
        self.model_outputs_dir = (
            os.path.join(self.root_outputs_dir, self.model_name)
        )
        
        self.checkpoint_path = (
            os.path.join(self.root_models_dir, self.model_checkpoint)
        )    
        self.text_embedder_dir = (
            os.path.join(self.clip_models_dir, "text_embedder/")
        )        
        self.text_embedder_path = (
            os.path.join(self.clip_models_dir, "text_embedder.safetensors")
        )
        self.tokenizer_path = (
            os.path.join(
                self.text_embedder_dir, "tokenizer/"
            )
        )
        self.text_model_path = (
            os.path.join(
                self.text_embedder_dir, self.clip_model_name
            )
        )

        self.image_encoder_path = (
            os.path.join(
                self.clip_models_dir, "image_encoder.safetensors"
            )
        )

        self.image_encoder_dir = (
            os.path.join(
                self.clip_models_dir, "image_encoder/"
            )
        )
        self.clip_model_path = (
            os.path.join(self.image_encoder_dir, "clip_model.ckpt")
        )
        self.image_processor_path = (
            os.path.join(
                self.image_encoder_dir, "image_processor.ckpt"
            )
        )
        self.unet_path = (
            os.path.join(self.sd_model_dir, "unet.safetensors")
        )
        self.autoencoder_path = (
            os.path.join(self.sd_model_dir, "autoencoder.safetensors")
        )
        self.encoder_path = (
            os.path.join(self.sd_model_dir, "encoder.safetensors")
        )
        self.decoder_path = (
            os.path.join(self.sd_model_dir, "decoder.safetensors")
        )
        self.latent_diffusion_path = (
            os.path.join(
                self.sd_model_dir, "latent_diffusion.safetensors"
            )
        )

    def __repr__(self):
        return f"""ModelsPathTree:
                {self.base_io_directory}/
                -- model's root dir: {self.root_models_dir}
                --- stable diffusion model checkpoint: {self.checkpoint_path}
                --- stable diffusion submodels dir: {self.sd_model_dir}
                ---- unet model: {self.unet_path}
                ---- autoencoder model: {self.autoencoder_path}
                ---- encoder model: {self.encoder_path}
                ---- decoder model: {self.decoder_path}
                ---- latent_diffusion model: {self.latent_diffusion_path}
                --- clip models' root dir: {self.clip_models_dir}
                ---- clip image encoder checkpoint path: {self.image_encoder_path}
                ---- clip text embedder checkpoint path: {self.text_embedder_path}
                ---- clip text embedder submodels dir: {self.text_embedder_dir}
                ----- tokenizer: {self.tokenizer_path}
                ----- text model: {self.text_model_path}
                ---- clip image encoder submodels dir: {self.image_encoder_dir}
                ----- clip_model: {self.clip_model_path}
                ----- image_processor: {self.image_processor_path}
                -- models' outputs root'dir: {self.root_outputs_dir}
                --- model's outputs dir: {self.model_outputs_dir}
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
    def text_model(self):
        """returns a dict {text_model_path: self.text_model_path}"""
        return {"text_model_path": self.text_model_path}

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
    ROOT_MODELS_DIR = (os.path.join(base_directory, ROOT_MODELS_PREFIX))

    CHECKPOINT_PATH = (
        os.path.join(ROOT_MODELS_DIR, CHECKPOINT)
    )
    TEXT_EMBEDDER_PATH = (
        os.path.join(ROOT_MODELS_DIR, "clip_text_embedder/clip_embedder.safetensors")
    )
    TOKENIZER_PATH = (
        os.path.join(ROOT_MODELS_DIR, "clip_text_embedder/clip_tokenizer.ckpt")
    )
    TEXT_MODEL_PATH = (
        os.path.join(ROOT_MODELS_DIR, "clip_text_embedder/clip_transformer.ckpt")
    )

    IMAGE_PROCESSOR_PATH = (
        os.path.join(ROOT_MODELS_DIR, "clip_image_encoder/image_processor.ckpt")
    )
    CLIP_MODEL_PATH = (
        os.path.join(ROOT_MODELS_DIR, "clip_image_encoder/clip_model.ckpt")
    )
    IMAGE_ENCODER_PATH = (
        os.path.join(ROOT_MODELS_DIR, "clip_image_encoder/clip_image_encoder.ckpt")
    )

    UNET_PATH = (os.path.join(ROOT_MODELS_DIR, "unet/unet.safetensors"))

    AUTOENCODER_PATH = (
        os.path.join(ROOT_MODELS_DIR, "autoencoder/autoencoder.safetensors")
    )
    ENCODER_PATH = (
        os.path.join(ROOT_MODELS_DIR, "autoencoder/encoder.safetensors")
    )
    DECODER_PATH = (
        os.path.join(ROOT_MODELS_DIR, "autoencoder/decoder.safetensors")
    )

    LATENT_DIFFUSION_PATH = (
        os.path.join(ROOT_MODELS_DIR, "latent_diffusion/latent_diffusion.safetensors")
    )

    paths_dict = {
        "root_models_path": ROOT_MODELS_DIR,
        "checkpoint_path": CHECKPOINT_PATH,
        "embedder_path": TEXT_EMBEDDER_PATH,
        "embedder_submodels_paths": {
            "tokenizer_path": TOKENIZER_PATH,
            "transformer_path": TEXT_MODEL_PATH,
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
            "embedder_path": TEXT_EMBEDDER_PATH,
            "unet_path": UNET_PATH,
            "autoencoder_path": AUTOENCODER_PATH,
        },
    }

    return paths_dict


def create_latent_diffusion_submodels_tree_paths(base_directory: str = "./"):
    ROOT_MODELS_DIR = (os.path.join(base_directory, ROOT_MODELS_PREFIX))

    TEXT_EMBEDDER_PATH = (
        os.path.join(ROOT_MODELS_DIR, "clip_text_embedder/clip_embedder.ckpt")
    )
    TOKENIZER_PATH = (
        os.path.join(ROOT_MODELS_DIR, "clip_text_embedder/clip_tokenizer.ckpt")
    )
    TEXT_MODEL_PATH = (
        os.path.join(ROOT_MODELS_DIR, "clip_text_embedder/clip_transformer.ckpt")
    )

    UNET_PATH = (os.path.join(ROOT_MODELS_DIR, "unet/unet.safetensors"))

    AUTOENCODER_PATH = (
        os.path.join(ROOT_MODELS_DIR, "autoencoder/autoencoder.safetensors")
    )
    ENCODER_PATH = (
        os.path.join(ROOT_MODELS_DIR, "autoencoder/encoder.safetensors")
    )
    DECODER_PATH = (
        os.path.join(ROOT_MODELS_DIR, "autoencoder/decoder.safetensors")
    )

    paths_dict = {
        "embedder_path": TEXT_EMBEDDER_PATH,
        "tokenizer_path": TOKENIZER_PATH,
        "transformer_path": TEXT_MODEL_PATH,
        "autoencoder_path": AUTOENCODER_PATH,
        "encoder_path": ENCODER_PATH,
        "decoder_path": DECODER_PATH,
        "unet_path": UNET_PATH,
    }

    return paths_dict
