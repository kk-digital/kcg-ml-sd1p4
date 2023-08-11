from configs.model_config import ModelPathConfig

config = ModelPathConfig(check_existence=False)


class CLIPconfigs:
    TEXT_EMBEDDER = 'clip/clip_txt_emb'
    TEXT_MODEL = 'clip/clip_txt_emb_model'
    TOKENIZER = 'clip/clip_txt_emb_tokenizer'

    IMAGE_ENCODER = 'clip/clip_img_enc'
    IMAGE_PROCESSOR = 'clip/clip_img_enc_processor'
    VISION_MODEL = 'clip/clip_img_enc_vision'
    CLIP_MODEL = 'clip/vit-large-patch14'


class SDconfigs:
    CHECKPOINT = 'sd/v1-5-pruned-emaonly'
    UNET = 'unet/unet'
    LATENT_DIFFUSION = 'sd/sd_latent_diffusion'

    AUTOENCODER = 'vae/vae'
    ENCODER = 'vae/vae_encoder'
    DECODER = 'vae/vae_decoder'


TEXT_EMBEDDER_PATH = config.get_model(CLIPconfigs.TEXT_EMBEDDER)
TOKENIZER_DIR_PATH = config.get_model_path(CLIPconfigs.TOKENIZER)
TEXT_MODEL_DIR_PATH = config.get_model_path(CLIPconfigs.TEXT_MODEL)

IMAGE_PROCESSOR_DIR_PATH = config.get_model_path(CLIPconfigs.IMAGE_PROCESSOR)
VISION_MODEL_DIR_PATH = config.get_model_path(CLIPconfigs.VISION_MODEL)
IMAGE_ENCODER_PATH = config.get_model(CLIPconfigs.IMAGE_ENCODER)

CHECKPOINT_PATH = config.get_model(SDconfigs.CHECKPOINT)
UNET_PATH = config.get_model(SDconfigs.UNET)
AUTOENCODER_PATH = config.get_model(SDconfigs.AUTOENCODER)
ENCODER_PATH = config.get_model(SDconfigs.ENCODER)
DECODER_PATH = config.get_model(SDconfigs.DECODER)
LATENT_DIFFUSION_PATH = config.get_model(SDconfigs.LATENT_DIFFUSION)


class IODirectoryTree:
    """returns dicts to be used as kwargs for loading submodels.
    the keys are the same as the kwargs used for the load methods."""

    def __init__(self, config_parm: ModelPathConfig):
        self.config = config_parm or config

    @property
    def checkpoint(self):
        """ "returns a dict {checkpoint_path: self.checkpoint_path}"""
        return self.config.get_model(SDconfigs.CHECKPOINT, to_dict=True)

    @property
    def encoder(self):
        """ "returns a dict {encoder_path: self.encoder_path}"""
        return self.config.get_model(SDconfigs.ENCODER, to_dict=True)

    @property
    def decoder(self):
        """returns a dict {decoder_path: self.decoder_path}"""
        return self.config.get_model(SDconfigs.DECODER, to_dict=True)

    @property
    def autoencoder(self):
        """returns a dict {autoencoder_path: self.autoencoder_path}"""
        return self.config.get_model(SDconfigs.AUTOENCODER, to_dict=True)

    @property
    def unet(self):
        """returns a dict {unet_path: self.unet_path}"""
        return self.config.get_model(SDconfigs.UNET, to_dict=True)

    @property
    def embedder(self):
        """returns a dict {embedder_path: self.embedder_path}"""
        return self.config.get_model(CLIPconfigs.TEXT_EMBEDDER, to_dict=True)

    @property
    def tokenizer(self):
        """returns a dict {tokenizer_path: self.tokenizer_path}"""
        return self.config.get_model(CLIPconfigs.TOKENIZER, to_dict=True)

    @property
    def text_model(self):
        """returns a dict {text_model_path: self.text_model_path}"""
        return self.config.get_model(CLIPconfigs.TEXT_MODEL, to_dict=True)

    @property
    def image_processor(self):
        """returns a dict {image_processor_path: self.image_processor_path}"""
        return self.config.get_model(CLIPconfigs.IMAGE_PROCESSOR, to_dict=True)

    @property
    def vision_model(self):
        """returns a dict {clip_model_path: self.clip_model_path}"""
        return self.config.get_model(CLIPconfigs.VISION_MODEL, to_dict=True)

    @property
    def image_encoder(self):
        return self.config.get_model(CLIPconfigs.IMAGE_ENCODER, to_dict=True)

    @property
    def latent_diffusion(self):
        return self.config.get_model(SDconfigs.LATENT_DIFFUSION, to_dict=True)

    @property
    def latent_diffusion_submodels(self):
        return self.config.get_model(CLIPconfigs.TEXT_EMBEDDER, SDconfigs.UNET, SDconfigs.AUTOENCODER, to_dict=True)

    @property
    def latent_diffusion_submodels_tree(self):
        return self.config.get_model(
            CLIPconfigs.TEXT_EMBEDDER,
            CLIPconfigs.TOKENIZER,
            CLIPconfigs.TEXT_MODEL,
            SDconfigs.UNET,
            SDconfigs.AUTOENCODER,
            SDconfigs.ENCODER,
            SDconfigs.DECODER,
            to_dict=True,
        )

    @property
    def embedder_submodels(self):
        return self.config.get_model(CLIPconfigs.TOKENIZER, CLIPconfigs.TEXT_MODEL, to_dict=True)

    @property
    def image_encoder_submodels(self):
        return self.config.get_model(CLIPconfigs.VISION_MODEL, CLIPconfigs.IMAGE_PROCESSOR, to_dict=True)

    @property
    def autoencoder_submodels(self):
        return self.config.get_model(SDconfigs.ENCODER, SDconfigs.DECODER, to_dict=True)
