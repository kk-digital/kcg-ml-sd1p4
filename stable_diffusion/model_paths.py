from configs.model_config import ModelPathConfig

config = ModelPathConfig(check_existence=False)


class CLIPconfigs:
    TXT_EMB = 'clip/txt_emb'
    TXT_EMB_TEXT_MODEL = 'clip/txt_emb_model'
    TXT_EMB_TOKENIZER = 'clip/txt_emb_tokenizer'

    IMG_ENC = 'clip/img_enc'
    IMG_ENC_PROCESSOR = 'clip/img_enc_processor'
    IMG_ENC_VISION = 'clip/img_enc_vision'

    CLIP_MODEL = 'clip/vit-large-patch14'


class SDconfigs:
    CHECKPOINT = 'sd/v1-5-pruned-emaonly'
    UNET = 'unet/unet'
    LATENT_DIFFUSION = 'sd/sd_latent_diffusion'

    VAE = 'vae/vae'
    VAE_ENCODER = 'vae/encoder'
    VAE_DECODER = 'vae/decoder'


CLIP_TEXT_EMBEDDER_PATH = config.get_model(CLIPconfigs.TXT_EMB)
CLIP_TOKENIZER_DIR_PATH = config.get_model_folder_path(CLIPconfigs.TXT_EMB_TOKENIZER)
CLIP_TEXT_MODEL_DIR_PATH = config.get_model_folder_path(CLIPconfigs.TXT_EMB_TEXT_MODEL)

CLIP_IMAGE_PROCESSOR_DIR_PATH = config.get_model_folder_path(CLIPconfigs.IMG_ENC_PROCESSOR)
CLIP_VISION_MODEL_DIR_PATH = config.get_model_folder_path(CLIPconfigs.IMG_ENC_VISION)
CLIP_IMAGE_ENCODER_PATH = config.get_model(CLIPconfigs.IMG_ENC)

CLIP_MODEL_PATH = config.get_model(CLIPconfigs.CLIP_MODEL)

SD_CHECKPOINT_PATH = config.get_model(SDconfigs.CHECKPOINT)
UNET_PATH = config.get_model(SDconfigs.UNET)
VAE_PATH = config.get_model(SDconfigs.VAE)
VAE_ENCODER_PATH = config.get_model(SDconfigs.VAE_ENCODER)
VAE_DECODER_PATH = config.get_model(SDconfigs.VAE_DECODER)
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
        return self.config.get_model(SDconfigs.VAE_ENCODER, to_dict=True)

    @property
    def decoder(self):
        """returns a dict {decoder_path: self.decoder_path}"""
        return self.config.get_model(SDconfigs.VAE_DECODER, to_dict=True)

    @property
    def autoencoder(self):
        """returns a dict {autoencoder_path: self.autoencoder_path}"""
        return self.config.get_model(SDconfigs.VAE, to_dict=True)

    @property
    def unet(self):
        """returns a dict {unet_path: self.unet_path}"""
        return self.config.get_model(SDconfigs.UNET, to_dict=True)

    @property
    def embedder(self):
        """returns a dict {embedder_path: self.embedder_path}"""
        return self.config.get_model(CLIPconfigs.TXT_EMB, to_dict=True)

    @property
    def tokenizer(self):
        """returns a dict {tokenizer_path: self.tokenizer_path}"""
        return self.config.get_model(CLIPconfigs.TXT_EMB_TOKENIZER, to_dict=True)

    @property
    def text_model(self):
        """returns a dict {text_model_path: self.text_model_path}"""
        return self.config.get_model(CLIPconfigs.TXT_EMB_TEXT_MODEL, to_dict=True)

    @property
    def image_processor(self):
        """returns a dict {image_processor_path: self.image_processor_path}"""
        return self.config.get_model(CLIPconfigs.IMG_ENC_PROCESSOR, to_dict=True)

    @property
    def vision_model(self):
        """returns a dict {clip_model_path: self.clip_model_path}"""
        return self.config.get_model(CLIPconfigs.IMG_ENC_VISION, to_dict=True)

    @property
    def image_encoder(self):
        return self.config.get_model(CLIPconfigs.IMG_ENC, to_dict=True)

    @property
    def latent_diffusion(self):
        return self.config.get_model(SDconfigs.LATENT_DIFFUSION, to_dict=True)

    @property
    def latent_diffusion_submodels(self):
        return self.config.get_model(CLIPconfigs.TXT_EMB, SDconfigs.UNET, SDconfigs.VAE, to_dict=True)

    @property
    def latent_diffusion_submodels_tree(self):
        return self.config.get_model(
            CLIPconfigs.TXT_EMB,
            CLIPconfigs.TXT_EMB_TOKENIZER,
            CLIPconfigs.TXT_EMB_TEXT_MODEL,
            SDconfigs.UNET,
            SDconfigs.VAE,
            SDconfigs.VAE_ENCODER,
            SDconfigs.VAE_DECODER,
            to_dict=True,
        )

    @property
    def embedder_submodels(self):
        return self.config.get_model(CLIPconfigs.TXT_EMB_TOKENIZER, CLIPconfigs.TXT_EMB_TEXT_MODEL, to_dict=True)

    @property
    def image_encoder_submodels(self):
        return self.config.get_model(CLIPconfigs.IMG_ENC_VISION, CLIPconfigs.IMG_ENC_PROCESSOR, to_dict=True)

    @property
    def autoencoder_submodels(self):
        return self.config.get_model(SDconfigs.VAE_ENCODER, SDconfigs.VAE_DECODER, to_dict=True)
