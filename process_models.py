from torchinfo import summary

from configs.model_config import ModelPathConfig
from stable_diffusion.model.clip_image_encoder import CLIPImageEncoder
from stable_diffusion.utils_model import initialize_latent_diffusion
from utility.labml.monit import section

config = ModelPathConfig()

if __name__ == "__main__":
    # CLIP
    with section(
            "Initialize CLIP image encoder and load submodels from lib"
    ):
        img_encoder = CLIPImageEncoder()
        img_encoder.load_submodels_from_transformer()
    with section("CLIP: Saving image encoder submodels"):
        img_encoder.save_submodels()
        img_encoder.unload_submodels()
        img_encoder.save()

    # STABLE DIFFUSION
    with section("Initializing Stable Diffusion Model"):
        model = initialize_latent_diffusion(
            path=config.get_model('sd/v1-5-pruned-emaonly'),
            force_submodels_init=True
        )
        summary(model)
    with section("Stable Diffusion: saving vae submodels"):
        model.first_stage_model.save_submodels()  # saves autoencoder submodels (encoder, decoder) with loaded state dict
    with section("Stable Diffusion: unloading vae submodels"):
        model.first_stage_model.unload_submodels()  # unloads autoencoder submodels
    with section("Stable Diffusion: saving embedder submodels"):
        model.cond_stage_model.save_submodels()  # saves text embedder submodels (tokenizer, transformer) with loaded state dict
    with section("Stable Diffusion: unloading embedder submodels"):
        model.cond_stage_model.unload_submodels()  # unloads text embedder submodels
    with section("Stable Diffusion: saving latent diffusion submodels"):
        model.save_submodels()  # saves latent diffusion submodels (autoencoder, clip_embedder, unet) with loaded state dict and unloaded submodels (when it applies)
    with section("Stable Diffusion: unloading latent diffusion submodels"):
        model.unload_submodels()  # unloads latent diffusion submodels
    with section("Stable Diffusion: saving latent diffusion model"):
        model.save()  # saves latent diffusion model with loaded state dict and unloaded submodels
