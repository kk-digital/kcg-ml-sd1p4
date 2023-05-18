import model.unet as unet
import model.clip_embedder as clip_embedder
import model.autoencoder as autoencoder

'''
unet_model: UNetModel,
                 autoencoder: Autoencoder,
                 clip_embedder: CLIPTextEmbedder,
                 latent_scaling_factor: float,
                 n_steps: int,
                 linear_start: float,
                 linear_end: float,
'''


class Config:
    UNET_PARAMS = {
        "image_size": 32,
        "in_channels": 4,
        "out_channels": 4,
        "model_channels": 320,
        "attention_resolutions": [
            4,
            2,
            1
        ],
        "num_res_blocks": 2,
        "channel_mult": [
            1,
            2,
            4,
            4
        ],
        "num_heads": 8,
        "use_spatial_transformer": True,
        "transformer_depth": 1,
        "context_dim": 768,
        "use_checkpoint": True,
        "legacy": False
    }

    AUTO_ENCODER_PARAMS = {
        "embed_dim": 4,
        "monitor": "val/rec_loss",
        "ddconfig": {
            "double_z": True,
            "z_channels": 4,
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": [
                1,
                2,
                4,
                4
            ],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0
        },
        "lossconfig": {
            "target": "torch.nn.Identity"
        }
    }


    @property
    def params(self):
        unet_model = unet.UNetModel(
            in_channels=self.UNET_PARAMS["in_channels"],
            out_channels=self.UNET_PARAMS["out_channels"],
            channels=self.UNET_PARAMS["model_channels"],
            n_res_blocks=self.UNET_PARAMS["num_res_blocks"],
            attention_levels=self.UNET_PARAMS["attention_resolutions"],
            channel_multipliers=self.UNET_PARAMS["channel_mult"],
            n_heads=self.UNET_PARAMS["num_heads"],
            tf_layers=self.UNET_PARAMS["transformer_depth"],
            d_cond=self.UNET_PARAMS["context_dim"],
        )

        autoencoder_model = autoencoder.Autoencoder(
            encoder=autoencoder.Encoder(
                channels=self.AUTO_ENCODER_PARAMS["ddconfig"]["ch"],
                channel_multipliers=self.AUTO_ENCODER_PARAMS["ddconfig"]["ch_mult"],
                n_resnet_blocks=self.AUTO_ENCODER_PARAMS["ddconfig"]["num_res_blocks"],
            ),
            decoder=autoencoder.Decoder(
                channels=self.AUTO_ENCODER_PARAMS["ddconfig"]["ch"],
                channel_multipliers=self.AUTO_ENCODER_PARAMS["ddconfig"]["ch_mult"],
                n_resnet_blocks=self.AUTO_ENCODER_PARAMS["ddconfig"]["num_res_blocks"],
                out_channels=self.AUTO_ENCODER_PARAMS["ddconfig"]["out_ch"],
                z_channels=self.AUTO_ENCODER_PARAMS["ddconfig"]["z_channels"],
            ),
            emb_channels=self.AUTO_ENCODER_PARAMS["embed_dim"],
            z_channels=self.AUTO_ENCODER_PARAMS["ddconfig"]["z_channels"],
        )

        clip_embedder_model = clip_embedder.CLIPTextEmbedder()

        return {
            "linear_start": 0.00085,
            "linear_end": 0.012,
            "n_steps": 1,
            "latent_scaling_factor": 0.18215,
            "unet_model": unet_model,
            "autoencoder": autoencoder_model,
            "clip_embedder": clip_embedder_model,
        }
