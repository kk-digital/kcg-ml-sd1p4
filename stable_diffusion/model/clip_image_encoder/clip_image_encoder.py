import os
import sys
import hashlib
import torch
import numpy as np
import PIL
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, Lambda
from torch import nn
import safetensors

from stable_diffusion.utils_logger import logger

sys.path.insert(0, os.getcwd())
from stable_diffusion.constants import IMAGE_PROCESSOR_PATH, VISION_MODEL_PATH, IMAGE_ENCODER_PATH
from stable_diffusion.utils_backend import get_device
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


class CLIPImageEncoder(nn.Module):

    def __init__(self, device=None, image_processor=None, vision_model=None):  # , input_mode = PIL.Image.Image):

        super().__init__()

        self.device = get_device(device)

        self.vision_model = vision_model
        self.image_processor = image_processor

        if image_processor is None:
            logger.warning(
                "image_processor has not been given. Initialize one with the `.initialize_preprocessor()` method.")
            # self.initialize_preprocessor()

        self.to(self.device)

    def save_submodels(self, image_processor_path=IMAGE_PROCESSOR_PATH, vision_model_path=VISION_MODEL_PATH):
        # Save the model to the specified folder
        self.image_processor.save_pretrained(image_processor_path)
        print("image_processor saved to: ", image_processor_path)
        self.vision_model.save_pretrained(vision_model_path, safe_serialization=True)
        print("vision_model saved to: ", vision_model_path)

    def load_submodels(self, image_processor_path=IMAGE_PROCESSOR_PATH, vision_model_path=VISION_MODEL_PATH):
        try:
            self.vision_model = (CLIPVisionModelWithProjection.from_pretrained(vision_model_path, local_files_only=True,
                                                                               use_safetensors=True)
                                 .eval()
                                 .to(self.device))
            logger.info(f"CLIPVisionModelWithProjection successfully loaded from : {vision_model_path}\n")
            self.image_processor = CLIPImageProcessor.from_pretrained(image_processor_path, local_files_only=True)
            logger.info(f"CLIPImageProcessor successfully loaded from : {image_processor_path}\n")
            return self
        except Exception as e:
            logger.error('Error loading submodels: ', e)

    def unload_submodels(self):
        # Unload the model from GPU memory
        if self.vision_model is not None:
            self.vision_model.to('cpu')
            del self.vision_model
            torch.cuda.empty_cache()
            self.vision_model = None
        if self.image_processor is not None:
            del self.image_processor
            torch.cuda.empty_cache()
            self.image_processor = None

    def save(self, image_encoder_path=IMAGE_ENCODER_PATH):
        try:
            safetensors.torch.save_model(self, image_encoder_path)
            print(f"CLIPImageEncoder saved to: {image_encoder_path}")
        except Exception as e:
            print(f"CLIPImageEncoder not saved. Error: {e}")

    def load(self, image_encoder_path: str = IMAGE_ENCODER_PATH):
        try:
            safetensors.torch.load_model(self, image_encoder_path, strict=True)
            print(f"CLIPTextEmbedder loaded from: {image_encoder_path}")
            return self
        except Exception as e:
            print(f"CLIPTextEmbedder not loaded. Error: {e}")

    def convert_image_to_tensor(self, image: PIL.Image.Image):
        return torch.from_numpy(np.array(image)) \
            .permute(2, 0, 1) \
            .unsqueeze(0) \
            .to(self.device) * (2 / 255.) - 1.0

    def preprocess_input(self, image):
        # Preprocess image
        if self.get_input_type(image) == PIL.Image.Image:
            image = (self.convert_image_to_tensor(image) + 1) / 2
        return self.image_processor(image, return_tensors="pt").to(self.device)

    def forward(self, image, do_preprocess=False):
        # Preprocess image
        # Compute CLIP features
        with torch.no_grad():
            if do_preprocess:
                processed_image = self.preprocess_input(image)
                features = self.vision_model(**processed_image).image_embeds
            else:
                features = self.vision_model(pixel_values=image).image_embeds
        return features

    @staticmethod
    def compute_sha256(image_data):
        # Compute SHA256
        return hashlib.sha256(image_data).hexdigest()

    @staticmethod
    def convert_image_to_rgb(image):
        return image.convert("RGB")

    @staticmethod
    def get_input_type(image):
        if isinstance(image, PIL.Image.Image):
            return PIL.Image.Image
        elif isinstance(image, torch.Tensor):
            return torch.Tensor
        else:
            raise ValueError("Image must be PIL Image or Tensor")

    def initialize_preprocessor(self, size=224, do_resize=True, do_center_crop=True, do_normalize=True):
        print("Initializing image preprocessor...")

        self.image_processor = Compose([
            Resize(size) if do_resize else Lambda(lambda x: x),
            CenterCrop(size) if do_center_crop else Lambda(lambda x: x),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)
            ) if do_normalize else Lambda(lambda x: x),
        ])
