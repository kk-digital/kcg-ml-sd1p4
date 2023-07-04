import torch
from stable_diffusion_base_script_custom import StableDiffusionBaseScript
from stable_diffusion.model.clip_embedder import CLIPTextEmbedder
from stable_diffusion.model.unet import UNetModel
from stable_diffusion.model.autoencoder import Encoder, Decoder, Autoencoder

# Initialize the U-Net
def initialize_unet(device = 'cpu', model_path = None) -> UNetModel:
    
    if model_path is None:
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
        unet_model = torch.load(model_path)
        unet_model.eval()
        return unet_model
        
# Initialize the autoencoder
def initialize_autoencoder(device = 'cpu', model_path = None) -> Autoencoder:

    if model_path is None:
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
        autoencoder = torch.load(model_path)
        autoencoder.eval()
        return autoencoder

# Initialize the CLIP text embedder
def initialize_clip_embedder(device = 'cpu', model_path = None) -> CLIPTextEmbedder:

    if model_path is None:
        clip_text_embedder = CLIPTextEmbedder(
            device=device,
        )
        torch.save(clip_text_embedder, './input/model/clip_embedder.ckpt')
        return clip_text_embedder
    else:
        clip_text_embedder = torch.load(model_path)
        clip_text_embedder.eval()
        return clip_text_embedder

# Summary Unet Model
def Summary_Unet_Model():
    print("Printing Unet Layers for: ConvNet(nn.Module)")
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

# Summary Autoencoder model
def Summary_Autoencoder_Model():
    
    print("Printing Autoencoder Layers for: ConvNet(nn.Module)")
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

# Summary CLIP Embedder model           
def Summary_Clip_Embedder_Model():
    
    print("Printing CLIP Embedder Layers for: ConvNet(nn.Module)")
    clip_text_embedder = CLIPTextEmbedder(
        device="cpu",
    )
    clip_text_embedder.PrintTorchInfo()

Summary_Unet_Model()
Summary_Autoencoder_Model()
Summary_Clip_Embedder_Model()