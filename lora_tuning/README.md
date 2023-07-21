# LoRa models for Stable Diffusion
Open up the notebook [here](../notebooks/lora_train.ipynb)

## Training a LoRa Stable Diffusion model
### Set-up
Install dependencies needed
```bash
apt update && apt install -y libgl1-mesa-glx git aria2
# Make sure you have the original base Stable Diffusion model under ./input/models!
pip3 install -r lora_tuning/requirements.txt
```
*Replace 'apt' with your package manager of choice*

### Let's get to actual training
```bash
python3 lora_tuning/lora_train.py --num_repeats 16
# Add --lowram if running on a system with less than 20GB of RAM
```
*Running this will use the default chibi waifu dataset to train a small LoRa*

## Generating images from the Stable Diffusion LoRa
### Set-up
```bash
pip3 install -r lora_tuning/inference/requirements.txt
```
***Very important! We need to run this, in order to install a newer version of diffusers (>17.0.1)***

### Generating images
```bash
python3 lora_tuning/inference/txt2img.py --checkpoint_path {checkpoint_path} --lora {lora_path} --output {output_filename} --scale {scale} --prompt {prompt-here}
```
Replace the variables in brakcets {} with the respectable files and directories

**This only supports square ratios for generating images as of now, improvements will be made if necessary**