import os

from torchinfo import summary

from configs.model_config import ModelPathConfig
from stable_diffusion.model.clip_image_encoder import CLIPImageEncoder
from stable_diffusion.utils_model import initialize_latent_diffusion
from utility.labml.monit import section

config = ModelPathConfig()

def check_model_path(path):
    """Check if the model file exists. If not, raise an error."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file {path} does not exist!")

if __name__ == "__main__":
    # CLIP
    with section("Initialize CLIP image encoder and load submodels from lib"):
        img_encoder = CLIPImageEncoder()
        img_encoder.load_submodels_from_transformer()
    with section("CLIP: Saving image encoder submodels"):
        img_encoder.save_submodels()
        img_encoder.unload_submodels()
        img_encoder.save()

    # Check the model path for STABLE DIFFUSION
    model_path = config.get_model('sd/v1-5-pruned-emaonly')
    check_model_path(model_path)
    
    # STABLE DIFFUSION
    with section("Initializing Stable Diffusion Model"):
        model = initialize_latent_diffusion(
            path=model_path,
            force_submodels_init=True
        )
        summary(model)
    with section("Stable Diffusion: saving vae submodels"):
        model.first_stage_model.save_submodels()  
    with section("Stable Diffusion: unloading vae submodels"):
        model.first_stage_model.unload_submodels()  
    with section("Stable Diffusion: saving embedder submodels"):
        model.cond_stage_model.save_submodels()  
    with section("Stable Diffusion: unloading embedder submodels"):
        model.cond_stage_model.unload_submodels()  
    with section("Stable Diffusion: saving latent diffusion submodels"):
        model.save_submodels()  
    with section("Stable Diffusion: unloading latent diffusion submodels"):
        model.unload_submodels()  
    with section("Stable Diffusion: saving latent diffusion model"):
        model.save()  
