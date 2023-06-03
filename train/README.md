# Training a LoRa using the notebook

## Setting up variables
Open up the notebook [here](../notebooks/lora_train.ipynb)

### What each variable does
```python3
project_name: Specifies the name of the directory to store the logs, output, config and dataset on by default
*_folder: directories on where to store certain files, stemming from project_name

optimizer: Specifies the optimizer to use for training the LoRa model.
optimizer_args: Additional arguments or configuration specific to the chosen optimizer.
continue_from_lora: File path to continue training from a saved LoRa model.
weighted_captions: Specifies whether to use weighted captions during training.
adjust_tags: Specifies whether to adjust the tags during training.
keep_tokens_weight: Weight parameter used for keeping tokens during training.

model_filename: Filename of the Stable Diffusion model to use for training.
model_file: File path of the Stable Diffusion model.

custom_model_is_based_on_sd2: Specifies whether the custom model is based on Stable Diffusion version 2.

resolution: Resolution of the dataset images.
flip_aug: Specifies whether to apply flip augmentation during training.
caption_extension: Extension of the caption files in the dataset.
activation_tags: Tags used for activation.
keep_tokens: Number of tokens to keep.
num_repeats: Number of repeats for training.
max_train_epochs: Maximum number of training epochs.
max_train_steps: Maximum number of training steps (if specified, it takes precedence over max_train_epochs).
save_every_n_epochs: Interval for saving the model during training (in terms of epochs).
keep_only_last_n_epochs: Number of most recent epochs to keep during training.

train_batch_size: Batch size for training.
unet_lr: Learning rate for the U-Net model.
text_encoder_lr: Learning rate for the text encoder model.
lr_scheduler: Learning rate scheduler to use during training.
lr_scheduler_num_cycles: Number of cycles for the learning rate scheduler.
lr_warmup_ratio: Warm-up ratio for the learning rate.
lr_warmup_steps: Number of warm-up steps for the learning rate.
min_snr_gamma_value: Minimum value for the signal-to-noise ratio gamma.

lora_type: Type of the LoRa model.
network_dim: Dimension of the network.
network_alpha: Alpha parameter used in the network.
network_module: Module for the LoRa network.
network_args: Additional arguments or configuration specific to the LoRa network.
```
