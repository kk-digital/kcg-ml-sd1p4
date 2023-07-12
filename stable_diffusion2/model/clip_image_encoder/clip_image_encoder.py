import os
import sys
sys.path.insert(0, os.getcwd())

import clip
import hashlib
import time
import torch
import tqdm

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from typing import Any, Union, List

from typing import List
from torch import nn, save
from os.path import join


from stable_diffusion2.constants import IMAGE_PROCESSOR_PATH, CLIP_MODEL_PATH, IMAGE_ENCODER_PATH
from stable_diffusion2.utils.utils import check_device
from torchinfo import summary


class CLIPImageEncoder(nn.Module):
    def __init__(self, device = None, image_processor = None, clip_model = None ):
        super().__init__()
        self.device = check_device(device)
    
        self.clip_model = clip_model
        self.image_processor = image_processor

        self.to(self.device)

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

    def forward(self, image: Image, batch_size: int = 1):
        # Preprocess image
        image = self.image_processor(image).unsqueeze(0).to(self.device)
        # Compute CLIP features
        with torch.no_grad():
            features = self.clip_model.encode_image(image)
        return features.cpu().numpy()

    @staticmethod
    def compute_sha256(image_data):
        # Compute SHA256
        return hashlib.sha256(image_data).hexdigest()

    @staticmethod
    def convert_file_to_tensor(image_data):
        # Convert file to tensor
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('RGB')  # Convert to RGB if needed
        image_tensor = torch.from_numpy(np.array(image))
        return image_tensor
    
    def convert_image_to_rgb(image):
        return image.convert("RGB")
    
    def initialize_preprocesser(self, size = 224, from_tensor = True):
        if from_tensor:
            self.image_processor = Compose([
                                    Resize(size),
                                    CenterCrop(size),
                                    Normalize(
                                        (0.48145466, 0.4578275, 0.40821073), 
                                        (0.26862954, 0.26130258, 0.27577711)
                                        ),
                                    ])     
        else:
            self.image_processor = Compose([
                                    Resize(size),
                                    CenterCrop(size),
                                    self.convert_image_to_rgb,
                                    ToTensor(),
                                    Normalize(
                                        (0.48145466, 0.4578275, 0.40821073), 
                                        (0.26862954, 0.26130258, 0.27577711)
                                        ),
                                    ])
    def preprocess_image(self, n_px):
        return Compose([
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            self.convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    def preprocess_image_tensor(self, n_px):
        return Compose([
            Resize(n_px),
            CenterCrop(n_px),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])        

    def compute_clip(self, image_data):
        # Compute clip for one image
        image_tensor = self.convert_file_to_tensor(image_data).to(self.device).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
        return features.cpu().numpy()

    def compute_clip_batch(self, images, batch_size):
        # Compute clip for a batch of images
        start_time = time.time()
        num_images = len(images)
        clip_features = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, num_images, batch_size):
                batch_images = images[i:i+batch_size]
                batch_images = [image.to(self.device) for image in batch_images]
                batch_features = self.model.encode_image(torch.stack(batch_images))
                clip_features.append(batch_features.cpu().numpy())
        clip_features = np.concatenate(clip_features, axis=0)
        end_time = time.time()
        print("Processed {} images in {:.2f} seconds. Speed: {:.2f} images/second, {:.2f} MB/second".format(
            num_images, end_time - start_time, num_images / (end_time - start_time),
            (num_images * images[0].nbytes / 1024 / 1024) / (end_time - start_time)))
        return clip_features
    
    def model_exists(root: str, filename: str):
        """
        Check if the model exists in the specified directory.
        """
        model_path = os.path.join(root, filename)
        return os.path.isfile(model_path)

    def verify_model(root: str, filename: str, expected_sha256: str):
        """
        Verify the SHA256 hash of the model file.
        """
        model_path = os.path.join(root, filename)
        if os.path.isfile(model_path):
            actual_sha256 = hashlib.sha256(open(model_path, "rb").read()).hexdigest()
            return actual_sha256 == expected_sha256
        else:
            return False
        

    




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

