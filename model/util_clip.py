import clip
import torch

from stable_diffusion.utils_backend import get_device


class UtilClip():
    def __init__(self, device=None):
        self.device = get_device(device)

    def load_model(self, model_name='ViT-L/14'):
        model, preprocess = clip.load(model_name, self.device)
        self.model = model
        self.preprocess = preprocess

    def unload_model(self):
        del self.model
        del self.processor

    def get_image_features(self, image):
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        # Encode the image
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)

        image_features = image_features.to(torch.float32)
        return image_features
