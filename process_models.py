import configparser
from torchinfo import summary

from stable_diffusion.model.clip_image_encoder import CLIPImageEncoder
from stable_diffusion.utils_model import initialize_latent_diffusion
from utility.labml.monit import section

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read('config.ini')

if __name__ == "__main__":
    with section("Initializing CLIP image encoder"):
        model = initialize_latent_diffusion(
            path=config["STABLE_DIFFUSION_PATHS"].get('CHECKPOINT_PATH'),
            force_submodels_init=True
        )
        summary(model)

    with section(
            "initialize CLIP image encoder and load submodels from lib"
    ):
        img_encoder = CLIPImageEncoder()
        img_encoder.load_submodels(image_processor_path=config["MODELS_DIRS"].get('CLIP_MODEL_DIR'),
                                   vision_model_path=config["MODELS_DIRS"].get('CLIP_MODEL_DIR'))
    with section("save image encoder submodels"):
        img_encoder.save_submodels()
        img_encoder.unload_submodels()
        img_encoder.save()
    with section("save vae submodels"):
        model.first_stage_model.save_submodels()  # saves autoencoder submodels (encoder, decoder) with loaded state dict
    with section("unload vae submodels"):
        model.first_stage_model.unload_submodels()  # unloads autoencoder submodels
    with section("save embedder submodels"):
        model.cond_stage_model.save_submodels()  # saves text embedder submodels (tokenizer, transformer) with loaded state dict
    with section("unload embedder submodels"):
        model.cond_stage_model.unload_submodels()  # unloads text embedder submodels
    with section("save latent diffusion submodels"):
        model.save_submodels()  # saves latent diffusion submodels (autoencoder, clip_embedder, unet) with loaded state dict and unloaded submodels (when it applies)
    with section("unload latent diffusion submodels"):
        model.unload_submodels()  # unloads latent diffusion submodels
    with section("save latent diffusion model"):
        model.save()  # saves latent diffusion model with loaded state dict and unloaded submodels