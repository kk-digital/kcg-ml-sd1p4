import torch
from stable_diffusion_base_script_custom import StableDiffusionBaseScript
from stable_diffusion.utils.model import save_images, set_seed, get_autocast
from stable_diffusion.model.unet_attention import CrossAttention
from stable_diffusion.latent_diffusion import LatentDiffusion
from stable_diffusion.model.clip_embedder import CLIPTextEmbedder
from stable_diffusion.model.unet import UNetModel
from labml import monit
from labml.logger import inspect
import os
from pathlib import Path
from typing import Union


from stable_diffusion.model.autoencoder import Encoder, Decoder, Autoencoder
from torchinfo import summary

encoder = Encoder(z_channels=4,
                          in_channels=3,
                          channels=128,
                          channel_multipliers=[1, 2, 4, 4],
                          n_resnet_blocks=2)

decoder = Decoder(out_channels=3,
                          z_channels=4,
                          channels=128,
                          channel_multipliers=[1, 2, 4, 4],
                          n_resnet_blocks=2)

model = Autoencoder(emb_channels=4,
                                  encoder=encoder,
                                  decoder=decoder,
                                  z_channels=4)

#torch.save(model, './input/model/autoencoder.ckpt')

def initialize_unet(device = 'cpu', model_path = None) -> UNetModel:
    
    # Initialize the U-Net
    if model_path is None:
        with monit.section('Initialize U-Net'):
            unet_model = UNetModel(in_channels=4,
                                out_channels=4,
                                channels=320,
                                attention_levels=[0, 1, 2],
                                n_res_blocks=2,
                                channel_multipliers=[1, 2, 4, 4],
                                n_heads=8,
                                tf_layers=1,
                                d_cond=768)
            torch.save(unet_model, './input/model/unet.ckpt')
            return unet_model
    else:
        with monit.section('Initialize U-Net'):
            unet_model = torch.load(model_path)
            unet_model.eval()
            return unet_model
        
def initialize_autoencoder(device = 'cpu', model_path = None) -> Autoencoder:

    # Initialize the autoencoder
    if model_path is None:
        with monit.section('Initialize autoencoder'):
            encoder = Encoder(z_channels=4,
                            in_channels=3,
                            channels=128,
                            channel_multipliers=[1, 2, 4, 4],
                            n_resnet_blocks=2)

            decoder = Decoder(out_channels=3,
                            z_channels=4,
                            channels=128,
                            channel_multipliers=[1, 2, 4, 4],
                            n_resnet_blocks=2)

            autoencoder = Autoencoder(emb_channels=4,
                                    encoder=encoder,
                                    decoder=decoder,
                                    z_channels=4)
            
            torch.save(autoencoder, './input/model/autoencoder.ckpt')
            
            return autoencoder
    else:
        with monit.section('Initialize autoencoder'):
            autoencoder = torch.load(model_path)
            autoencoder.eval()
            return autoencoder

def initialize_clip_embedder(device = 'cpu', model_path = None) -> CLIPTextEmbedder:

    # Initialize the CLIP text embedder
    if model_path is None:
        with monit.section('Initialize CLIP Embedder'):
            clip_text_embedder = CLIPTextEmbedder(
                device=device,
            )
            torch.save(clip_text_embedder, './input/model/clip_embedder.ckpt')
            return clip_text_embedder
    else:
        with monit.section('Initialize CLIP Embedder'):
            clip_text_embedder = torch.load(model_path)
            clip_text_embedder.eval()
            return clip_text_embedder
        
def load_model(path: Union[str, Path] = '', device = 'cpu', autoencoder = None, clip_text_embedder = None, unet_model = None) -> LatentDiffusion:
    """
    ### Load [`LatentDiffusion` model](latent_diffusion.html)
    """

    
    autoencoder = initialize_autoencoder(device=device, model_path=autoencoder)
    clip_text_embedder = initialize_clip_embedder(device=device, model_path=clip_text_embedder)
    unet_model = initialize_unet(device=device, model_path=unet_model)

    # Initialize the Latent Diffusion model
    with monit.section('Initialize Latent Diffusion model'):
        model = LatentDiffusion(linear_start=0.00085,
                                linear_end=0.0120,
                                n_steps=1000,
                                latent_scaling_factor=0.18215,

                                autoencoder=autoencoder,
                                clip_embedder=clip_text_embedder,
                                unet_model=unet_model)

    batch_size = 16
    summary(model,input_size=(batch_size,1, 28, 28), device="cpu")
    #summary(model, [(1, 16, 16), (1, 28, 28), (1, 28, 28) , (1, 28, 28)], device="cpu")
    return model

def Summary_Unet_Model():
    # Summary Unet Model
    print("Printing Unet Layers for: ConvNet(nn.Module)")
    with monit.section('Initialize U-Net'):
        unet_model = UNetModel(in_channels=4,
                            out_channels=4,
                            channels=320,
                            attention_levels=[0, 1, 2],
                            n_res_blocks=2,
                            channel_multipliers=[1, 2, 4, 4],
                            n_heads=8,
                            tf_layers=1,
                            d_cond=768)
        unet_model.PrintTorchInfo()

def Summary_Autoencoder_Model():
    # Autoencoder model
    print("Printing Autoencoder Layers for: ConvNet(nn.Module)")
    with monit.section('Initialize autoencoder'):
            encoder = Encoder(z_channels=4,
                            in_channels=3,
                            channels=128,
                            channel_multipliers=[1, 2, 4, 4],
                            n_resnet_blocks=2)

            decoder = Decoder(out_channels=3,
                            z_channels=4,
                            channels=128,
                            channel_multipliers=[1, 2, 4, 4],
                            n_resnet_blocks=2)

            autoencoder = Autoencoder(emb_channels=4,
                                    encoder=encoder,
                                    decoder=decoder,
                                    z_channels=4)
            autoencoder.PrintTorchInfo()
            
def Summary_Clip_Embedder_Model():
    # CLIP Embedder model
    print("Printing CLIP Embedder Layers for: ConvNet(nn.Module)")
    with monit.section('Initialize CLIP Embedder'):
        clip_text_embedder = CLIPTextEmbedder(
            device="cpu",
        )
        clip_text_embedder.PrintTorchInfo()

Summary_Unet_Model()
Summary_Autoencoder_Model()
Summary_Clip_Embedder_Model()
