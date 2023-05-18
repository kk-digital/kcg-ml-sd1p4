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

    LATENT_DIFFUSION_PARAMS = {
        "linear_start": 0.00085,
        "linear_end": 0.012,
        "n_steps": 1,
        "latent_scaling_factor": 0.18215,
    }

