import os
import sys
import configparser

BASE_DIRECTORY = os.getcwd()
# BASE_DIRECTORY = '../'

sys.path.insert(0, BASE_DIRECTORY)

# config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
# config.read("./config.ini")

# base = config["BASE"]
# root_dirs = config["ROOT_DIRS"]
# models_dirs = config["MODELS_DIRS"]
# submodels_dirs = config["SUBMODELS_DIRS"]
# stable_diffusion_paths = config['STABLE_DIFFUSION_PATHS']

BASE_IO_DIRECTORY = "./"
BASE_IO_DIRECTORY_PREFIX = ""
ROOT_MODELS_PREFIX = "input/model/"
ROOT_OUTPUTS_PREFIX = "output/model/"
MODEL = "v1-5-pruned-emaonly"
# MODEL = base.get("model_name")
CLIP_MODEL = "vit-large-patch14"

CHECKPOINT = f"{MODEL}.safetensors"
# CHECKPOINT_PATH = stable_diffusion_paths.get("checkpoint_path")

ROOT_MODELS_DIR = (os.path.join(BASE_IO_DIRECTORY, ROOT_MODELS_PREFIX))
# ROOT_MODELS_DIR = root_dirs.get("root_models_dir")
# ROOT_OUTPUTS_DIR = root_dirs.get("root_outputs_dir")
ROOT_OUTPUTS_DIR = (os.path.join(BASE_IO_DIRECTORY, ROOT_OUTPUTS_PREFIX))
SD_DEFAULT_MODEL_DIR = os.path.join(ROOT_MODELS_DIR, MODEL)
CLIP_MODELS_DIR = os.path.join(ROOT_MODELS_DIR, "clip")
TEXT_EMBEDDER_DIR = (
    os.path.join(CLIP_MODELS_DIR, "text_embedder")
)
IMAGE_ENCODER_DIR = (
    os.path.join(CLIP_MODELS_DIR, "image_encoder")
)
CHECKPOINT_PATH = (
    os.path.join(ROOT_MODELS_DIR, CHECKPOINT)
)
# SD_DEFAULT_MODEL_DIR = models_dirs.get("sd_default_model_dir")
# CLIP_MODELS_DIR = models_dirs.get("clip_models_dir")
# TEXT_EMBEDDER_DIR = models_dirs.get("text_embedder_dir")
# IMAGE_ENCODER_DIR = models_dirs.get("image_encoder_dir")
# CLIP_MODEL_DIR = models_dirs.get("clip_model_dir")
CLIP_MODEL_DIR = os.path.join(CLIP_MODELS_DIR, CLIP_MODEL)
TEXT_EMBEDDER_PATH = (
    os.path.join(TEXT_EMBEDDER_DIR, "text_embedder.safetensors")
)
TOKENIZER_PATH = (
    os.path.join(TEXT_EMBEDDER_DIR, "tokenizer")
)
TEXT_MODEL_PATH = (
    os.path.join(TEXT_EMBEDDER_DIR, 'text_model')
)

IMAGE_PROCESSOR_PATH = (
    os.path.join(IMAGE_ENCODER_DIR, "image_processor")
)
VISION_MODEL_PATH = (
    os.path.join(IMAGE_ENCODER_DIR, "vision_model")
)
IMAGE_ENCODER_PATH = (
    os.path.join(IMAGE_ENCODER_DIR, "image_encoder.safetensors")
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

# def create_directory_tree_paths(base_io_directory: str = BASE_IO_DIRECTORY, base_directory: str = BASE_DIRECTORY, model_name = MODEL, clip_model_name = CLIP_MODEL, root_models_prefix: str = ROOT_MODELS_PREFIX, root_outputs_prefix: str = ROOT_OUTPUTS_PREFIX, checkpoint_format = ".safetensors"):
def create_directory_tree_folders(config: configparser.ConfigParser):


    for k, v in config["ROOT_DIRS"].items():
        os.makedirs(v, exist_ok=True)    
    for k, v in config["MODELS_DIRS"].items():
        os.makedirs(v, exist_ok=True)
    for k, v in config["SUBMODELS_DIRS"].items():
        os.makedirs(v, exist_ok=True)

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
            "clip_model_path": VISION_MODEL_PATH,
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


class IODirectoryTree:
    """returns dicts to be used as kwargs for loading submodels.
    the keys are the same as the kwargs used for the load methods."""

    def __init__(self, 
                base_io_directory_prefix: str = "", 
                base_directory: str = "./", 
                model_name = "v1-5-pruned-emaonly", 
                clip_model_name = "vit-large-patch14", 
                root_models_prefix: str = "input/model/", 
                root_outputs_prefix: str = "output/model/", 
                checkpoint_format = ".safetensors"):
        
        self.base_directory = base_directory
        self.base_io_directory_prefix = base_io_directory_prefix
        self.base_io_directory = os.path.join(base_directory, base_io_directory_prefix)

        self.checkpoint_format = checkpoint_format

        self.root_models_prefix = root_models_prefix
        self.root_outputs_prefix = root_outputs_prefix

        self.model_name = model_name
        self.model_checkpoint = f"{self.model_name}{self.checkpoint_format}"

        self.clip_model_name = clip_model_name
        
        self.root_dirs = []
        
        self.root_models_dir = (
            os.path.join(self.base_io_directory, root_models_prefix)
        )
        self.root_dirs.append(self.root_models_dir)
        self.root_outputs_dir = (
            os.path.join(self.base_io_directory, root_outputs_prefix)
            )
        self.root_dirs.append(self.root_outputs_dir)
        self.models_dirs = []        
        self.sd_model_dir = (
            os.path.join(self.root_models_dir, self.model_name)
        )
        self.models_dirs.append(self.sd_model_dir)
        self.clip_models_dir = (
            os.path.join(self.root_models_dir, "clip")
        )
        self.models_dirs.append(self.clip_models_dir)
        self.text_embedder_dir = (
            os.path.join(self.clip_models_dir, "text_embedder")
        )
        self.models_dirs.append(self.text_embedder_dir)
        
        self.image_encoder_dir = (
            os.path.join(
                self.clip_models_dir, "image_encoder"
            )
        )        
        self.models_dirs.append(self.image_encoder_dir)
        self.model_outputs_dir = (
            os.path.join(self.root_outputs_dir, self.model_name)
        )
        self.models_dirs.append(self.model_outputs_dir)
                
        self.submodels_dirs = []
            
        self.tokenizer_path = (
            os.path.join(
                self.text_embedder_dir, "tokenizer"
            )
        )
        self.submodels_dirs.append(self.tokenizer_path)
        
        self.text_model_path = (
            os.path.join(
                self.text_embedder_dir, 'text_model'
            )
        )        
        
        self.submodels_dirs.append(self.text_model_path)
        
        self.text_embedder_path = (
            os.path.join(self.clip_models_dir, "text_embedder.safetensors")
        )
        

        self.image_encoder_path = (
            os.path.join(
                self.clip_models_dir, "image_encoder.safetensors"
            )
        )
        self.vision_model_path = (
            os.path.join(self.image_encoder_dir, "vision_model")
        )
        self.submodels_dirs.append(self.vision_model_path)
        self.image_processor_path = (
            os.path.join(
                self.image_encoder_dir, "image_processor"
            )
        )
        self.submodels_dirs.append(self.image_processor_path)
        
        self.checkpoint_path = (
            os.path.join(self.root_models_dir, self.model_checkpoint)
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
        return f"""IODirectoryTree:
                - {self.base_io_directory}
                -- model's root dir: {self.root_models_dir}
                --- stable diffusion model checkpoint: {self.checkpoint_path}
                --- stable diffusion submodels dir: {self.sd_model_dir}
                ---- unet model: {self.unet_path}
                ---- autoencoder model: {self.autoencoder_path}
                ---- encoder model: {self.encoder_path}
                ---- decoder model: {self.decoder_path}
                ---- latent_diffusion model: {self.latent_diffusion_path}
                --- clip models' root dir: {self.clip_models_dir}
                ---- clip text embedder submodels dir: {self.text_embedder_dir}
                ----- clip text embedder checkpoint path: {self.text_embedder_path}
                ----- tokenizer: {self.tokenizer_path}
                ----- text model: {self.text_model_path}
                ---- clip image encoder submodels dir: {self.image_encoder_dir}
                ----- clip image encoder checkpoint path: {self.image_encoder_path}
                ----- vision_model: {self.vision_model_path}
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
        return {"embedder_path": self.text_embedder_path}

    @property
    def tokenizer(self):
        """returns a dict {tokenizer_path: self.tokenizer_path}"""
        return {"tokenizer_path": self.tokenizer_path}

    @property
    def text_model(self):
        """returns a dict {text_model_path: self.text_model_path}"""
        return {"transformer_path": self.text_model_path}

    @property
    def image_processor(self):
        """returns a dict {image_processor_path: self.image_processor_path}"""
        return {"image_processor_path": self.image_processor_path}

    @property
    def vision_model(self):
        """returns a dict {clip_model_path: self.clip_model_path}"""
        return {"vision_model_path": self.vision_model_path}

    @property
    def image_encoder(self):
        return {"image_encoder_path": self.image_encoder_path}

    @property
    def latent_diffusion(self):
        return {"latent_diffusion_path": self.latent_diffusion_path}

    @property
    def latent_diffusion_submodels(self):
        return {
            "embedder_path": self.text_embedder_path,
            "unet_path": self.unet_path,
            "autoencoder_path": self.autoencoder_path,
        }

    @property
    def latent_diffusion_submodels_tree(self):
        return {
            "embedder_path": self.text_embedder_path,
            "tokenizer_path": self.tokenizer_path,
            "transformer_path": self.text_model_path,
            "unet_path": self.unet_path,
            "autoencoder_path": self.autoencoder_path,
            "encoder_path": self.encoder_path,
            "decoder_path": self.decoder_path,
        }

    @property
    def embedder_submodels(self):
        return {
            "tokenizer_path": self.tokenizer_path,
            "transformer_path": self.text_model_path,
        }

    @property
    def image_encoder_submodels(self):
        return {
            "vision_model_path": self.vision_model_path,
            "image_processor_path": self.image_processor_path,
        }

    @property
    def autoencoder_submodels(self):
        return {"encoder_path": self.encoder_path, "decoder_path": self.decoder_path}

    def create_directory_tree_folders(self):


        for v in self.__getattribute__("ROOT_DIRS".lower()):
            os.makedirs(v, exist_ok=True)    
        for v in self.__getattribute__("MODELS_DIRS".lower()):
            os.makedirs(v, exist_ok=True)
        for v in self.__getattribute__("SUBMODELS_DIRS".lower()):
            os.makedirs(v, exist_ok=True)

        paths_dict = {
            "root_models_path": self.root_models_dir,
            "checkpoint_path": self.checkpoint_path,
            "embedder_path": self.text_embedder_path,
            "embedder_submodels_paths": {
                "tokenizer_path": self.tokenizer_path,
                "transformer_path": self.text_model_path,
            },
            "image_encoder_path": self.image_encoder_path,
            "image_encoder_submodels_paths": {
                "image_processor_path": self.image_processor_path,
                "clip_model_path": self.vision_model_path,
            },
            "unet_path": self.unet_path,
            "autoencoder_path": self.autoencoder_path,
            "autoencoder_submodels_paths": {
                "encoder_path": self.encoder_path,
                "decoder_path": self.decoder_path,
            },
            "latent_diffusion_path": self.latent_diffusion_path,
            "latent_diffusion_submodels_paths": {
                "embedder_path": self.text_embedder_path,
                "unet_path": self.unet_path,
                "autoencoder_path": self.autoencoder_path,
            },
        }
        return paths_dict
