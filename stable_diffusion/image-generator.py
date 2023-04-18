from PIL import Image, ImageDraw, ImageFont
import io
from flask import Flask, request
import os

app = Flask(__name__)


@app.before_first_request
def load_models():
    os.system('pip install -r requirements.txt')

    import transformers
    import scipy
    import ftfy
    import accelerate
    import torch


    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler

    # what is function is PNDMScheduler??
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
    schenduler = LMSDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")



@app.route('/generate_image', methods=['POST'])
def generate_image():
    prompt = ["a photograph of a monkey riding a horse"]


    width = 512
    height = 512
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype("arial.ttf", 24)
    draw.text((10, 10), prompt, fill=(0, 0, 0))

    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    return img_bytes.getvalue(), 200


@app.route('/ping', methods=['GET'])
def ping():
    return 'Pong!', 200


if __name__ == '__main__':
    app.run(debug=True)
