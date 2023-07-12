import time
import zipfile
import hashlib
from PIL import Image
import io
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import warnings
from clip import load
import numpy as np
import os
import urllib
import tqdm
from typing import Any, Union, List
from pkg_resources import packaging


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

__all__ = ["available_models", "load", "tokenize"]
if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}

class ClipImageEncoder:
    def __init__(self, model_path):
        # Set the model folder path
        self.model_path = model_path
        self.model = None
        self.transform = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            print("Warning: CUDA is not available. Using CPU instead.")

    def available_models() -> List[str]:
        """Returns the names of available CLIP models"""
        return list(_MODELS.keys())        

    def check_and_download_model(self):
            # Check if the model exists in the provided model folder path
            if not os.path.exists(os.path.join(self.model_path, '')):
                # Download the model if it doesn't exist
                # Currently just prints a warning, replace with actual download logic if available
                print("Model not found. Downloading it...")
                # Code to download model goes here...

    def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
        """Load a CLIP model

        Parameters
        ----------
        name : str
            A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

        device : Union[str, torch.device]
            The device to put the loaded model

        jit : bool
            Whether to load the optimized JIT model or more hackable non-JIT model (default).

        download_root: str
            path to download the model files; by default, it uses "~/.cache/clip"

        Returns
        -------
        model : torch.nn.Module
            The CLIP model

        preprocess : Callable[[PIL.Image], torch.Tensor]
            A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
        """
        if name in _MODELS:
            model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
        elif os.path.isfile(name):
            model_path = name
        else:
            raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

        with open(model_path, 'rb') as opened_file:
            try:
                # loading JIT archive
                model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
                state_dict = None
            except RuntimeError:
                # loading saved state dict
                if jit:
                    warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                    jit = False
                state_dict = torch.load(opened_file, map_location="cpu")

        if not jit:
            model = build_model(state_dict or model.state_dict()).to(device)
            if str(device) == "cpu":
                model.float()
            return model, _transform(model.visual.input_resolution)


    def unload_model(self):
        # Unload the model from GPU memory
        del self.tranformer
        del self.model
        torch.cuda.empty_cache()

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
    
    def _convert_image_to_rgb(image):
        return image.convert("RGB")
        
    def ImageClipPreprocessor(n_px):
        return Compose([
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
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
        
    def download_model(url: str, root: str):
        """
        Download the model from the specified URL to the specified directory.
        """
        os.makedirs(root, exist_ok=True)
        filename = os.path.basename(url)
        expected_sha256 = url.split("/")[-2]
        download_target = os.path.join(root, filename)

        with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
            with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break
                    output.write(buffer)
                    loop.update(len(buffer))

        if not verify_model(root, filename, expected_sha256):
            raise RuntimeError("Model has been downloaded but the SHA256 checksum does not match")

        return download_target
    




encoder = ClipImageEncoder(model_path='path_to_model_folder')
encoder.load_model()

with zipfile.ZipFile("zipfile.zip", "r") as f:
    image_data_list = [f.read(name) for name in f.namelist() if name.lower().endswith(('.png', '.jpg', '.jpeg','.gif'))]

# Compute SHA256
sha256_list = [encoder.compute_sha256(image_data) for image_data in image_data_list]

# Convert image data to tensors
image_tensors = [encoder.convert_file_to_tensor(image_data) for image_data in image_data_list]

# Compute CLIP features for a batch
batch_size = 8  # Adjust the batch size as needed
clip_features = encoder.compute_clip_batch(image_tensors, batch_size)

# Unload model
encoder.unload_model()

