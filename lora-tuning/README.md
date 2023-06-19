# Training a LoRa using a script

## Usage
`lora_train.py` will train a LoRa, provided a tagged image dataset, a stable diffusion model, and parameters to create the network.

The script, by default, will train a LoRa with a sample dataset (of Waifus) gathered from pinterest. It will output a series of LoRa models for each `n` epochs as a `.safetensor` file, using a standard naming convention derived from the parameters that are used.
### Command line arguments
* --dataset: Path to the dataset directory or ZIP file. Default: ./test-images/chibi-waifu-pixelart.zip
* --repo_dir: Directory to clone the sd-scripts repository. Default: None
* --activation_tags: The number of activation tags in each txt file on the dataset. Default: 1
* --num_repeats: Number of times to repeat per image. Default: 10
* --max_train_epochs: How many epochs to train for. Default: 10
* --save_every_n_epochs: How frequently should we save the LoRa model. Default: 1
* --config_file: Path to the training configuration file. Default: None
* --config_dir: Path to store all of the generated config files. Default: None
* --dataset_config_file: Path to the dataset configuration file. Default: None
* --model_file: Path to the model file. Default: ./input/model/sd-v1-4.ckpt
* --log_dir: Path to store log files. Default: None
* --output_dir: Path to store output (LoRa model) files. Default: None
* --unet_lr: Learning rate for the UNet model. Default: 0.0005
* --text_encoder_lr: Learning rate for the text encoder model. Default: 0.0001
* --network_dim: Dimension of the network. Default: 512
* --network_alpha: Alpha value for the network. Default: None
* --batch_size: Number of images to use per epoch. Default: 3
* --caption_extension: Do not specify if there are no captions for the image. If there are, specify the extension of the captions (e.g., txt) here. Default: None
* --resolution: Resolution of the images. Must be a square aspect ratio (1:1). Default: 512
* --project_name: Put the project name here. Will dictate the filenames of the LoRa models produced, amongst other things. Default: "Test"
* --continue_from_lora: Path to the LoRa file from which to continue training. Default: None
* --accelerate_config_file: Path to the accelerate distributed training configuration file. Default: None

**If you need a more detailed description of the arguments run `python3 train/lora_train.py --help`**

## Acknowledgements
This script is based on the LoRa implementation by Kohya, and their [sd-scripts](https://github.com/kohya-ss/sd-scripts/) repository.
