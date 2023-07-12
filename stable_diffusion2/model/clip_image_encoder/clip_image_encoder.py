import os
import sys
sys.path.insert(0, os.getcwd())

import clip
import hashlib
import time
import torch
import numpy as np
import tqdm
import PIL
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Lambda
from typing import Any, Union, List

from typing import List
from torch import nn, save
from os.path import join


from stable_diffusion2.constants import IMAGE_PROCESSOR_PATH, CLIP_MODEL_PATH, IMAGE_ENCODER_PATH
from stable_diffusion2.utils.utils import check_device
from torchinfo import summary


class CLIPImageEncoder(nn.Module):

    def __init__(self, device = None, image_processor = None, clip_model = None):#, input_mode = PIL.Image.Image):

        super().__init__()

        self.device = check_device(device)

        self.clip_model = clip_model
        # self._image_processor = image_processor
        self.image_processor = image_processor
        # self.input_mode = input_mode

        if image_processor is None:
            print("WARNING: image_processor has not been given. Initialize one with the `.initialize_preprocessor()` method.")
            # self.initialize_preprocessor()

        self.to(self.device)


    # @property
    # def image_processor(self):
    #     if self.input_mode == PIL.Image.Image:
    #         return self._image_processor_image
    #     elif self.input_mode == torch.Tensor:
    #         return self._image_processor_tensor
    # @image_processor.setter
    # def image_processor(self, value):
    #     self._image_processor_image = value
    #     self._image_processor_tensor = value
    #     print(f"WARNING: image_processor has been set externally. This class image processer currently expects a {self.input_mode} as input. Please set the `input_mode` and provide inputs accordingly.")

    def load_from_lib(self, vit_model = 'ViT-L/14'):
        self.clip_model, self.image_processor = clip.load(vit_model, device=self.device)

    def load_submodels(self, image_processor_path = IMAGE_PROCESSOR_PATH, clip_model_path = CLIP_MODEL_PATH):

        self.image_processor = torch.load(image_processor_path, map_location=self.device)
        self.clip_model = torch.load(clip_model_path, map_location=self.device)
        self.clip_model.eval()

    def unload_submodels(self):
        # Unload the model from GPU memory
        del self.image_processor
        del self.clip_model
        torch.cuda.empty_cache()
        self.image_processor = None
        self.clip_model = None        

    def load_image_processor(self, image_processor_path = IMAGE_PROCESSOR_PATH):
        self.image_processor = torch.load(image_processor_path, map_location=self.device)
        return self.image_processor

    def load_clip_model(self, clip_model_path = CLIP_MODEL_PATH):

        self.clip_model = torch.load(clip_model_path, map_location=self.device)
        self.clip_model.eval()
        return self.clip_model

    def unload_image_processor(self):
        del self.image_processor
        torch.cuda.empty_cache()
        self.image_processor = None

    def unload_clip_model(self):
        del self.clip_model
        torch.cuda.empty_cache()
        self.clip_model = None

    def save(self, image_encoder_path = IMAGE_ENCODER_PATH):
        torch.save(self, image_encoder_path)

    def save_submodels(self, image_processor_path = IMAGE_PROCESSOR_PATH, clip_model_path = CLIP_MODEL_PATH):
        # Save the model to the specified folder
        torch.save(self.image_processor, image_processor_path)
        torch.save(self.clip_model, clip_model_path)
    
    def convert_image_to_tensor(self, image: PIL.Image.Image):
        return torch.from_numpy(np.array(image)) \
            .permute(2, 0, 1) \
            .unsqueeze(0) \
            .to(self.device) * (2/255.) - 1.0

    def preprocess_input(self, image: Union[PIL.Image.Image, torch.Tensor]):
        # Preprocess image
        if self.get_input_type(image) == PIL.Image.Image:
            image = self.convert_image_to_tensor(image)
        return self.image_processor(image).to(self.device)
    
    def forward(self, image: Union[PIL.Image.Image, torch.Tensor], do_preprocess = False):
        # Preprocess image
        if do_preprocess:
            image = self.preprocess_input(image)
        # Compute CLIP features
        with torch.no_grad():
            features = self.clip_model.encode_image(image)
        return features

    # def preprocess_input(self, image: Union[Image.Image, torch.Tensor]):
    #     # Preprocess image
    #     self.input_mode = self.get_input_type(image)
    #     return self.image_processor(image).to(self.device)
    
    # def forward(self, image: Union[PIL.Image.Image, torch.Tensor]):
    #     # Preprocess image
    #     image = self.preprocess_input(image)
    #     # Compute CLIP features
    #     with torch.no_grad():
    #         features = self.clip_model.encode_image(image)
    #     return features
    
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
        
    def initialize_preprocessor(self, size = 224):
        self.image_processor = Compose([
                                Resize(size),
                                CenterCrop(size),
                                Normalize(
                                    (0.48145466, 0.4578275, 0.40821073), 
                                    (0.26862954, 0.26130258, 0.27577711)
                                    ),
                                ])     
        
        # self._image_processor_tensor = Compose([
        #                         Resize(size),
        #                         CenterCrop(size),
        #                         Normalize(
        #                             (0.48145466, 0.4578275, 0.40821073), 
        #                             (0.26862954, 0.26130258, 0.27577711)
        #                             ),
        #                         ])     
        # self._image_processor_image = Compose([
        #                         Resize(size),
        #                         CenterCrop(size),
        #                         self.convert_image_to_rgb,
        #                         ToTensor(),
        #                         Lambda(lambda x: x * (2/255) - 1.0),
        #                         Normalize(
        #                             (0.48145466, 0.4578275, 0.40821073), 
        #                             (0.26862954, 0.26130258, 0.27577711)
        #                             ),
        #                         Lambda(lambda x: x.unsqueeze(0)),
        #                         ])

    # def verify_model(root: str, filename: str, expected_sha256: str):
    #     """
    #     Verify the SHA256 hash of the model file.
    #     """
    #     model_path = os.path.join(root, filename)
    #     if os.path.isfile(model_path):
    #         actual_sha256 = hashlib.sha256(open(model_path, "rb").read()).hexdigest()
    #         return actual_sha256 == expected_sha256
    #     else:
    #         return False
        

    




# encoder = CLIPImageEncoder(model_path='path_to_model_folder')
# encoder.load_model()

# with zipfile.ZipFile("zipfile.zip", "r") as f:
#     image_data_list = [f.read(name) for name in f.namelist() if name.lower().endswith(('.png', '.jpg', '.jpeg','.gif'))]

# # Compute SHA256
# sha256_list = [encoder.compute_sha256(image_data) for image_data in image_data_list]

# # Convert image data to tensors
# image_tensors = [encoder.convert_file_to_tensor(image_data) for image_data in image_data_list]

# # Compute CLIP features for a batch
# batch_size = 8  # Adjust the batch size as needed
# clip_features = encoder.compute_clip_batch(image_tensors, batch_size)

# # Unload model
# encoder.unload_model()

