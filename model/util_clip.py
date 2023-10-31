import clip
import torch
import torchvision.transforms as transforms

from stable_diffusion.utils_backend import get_device


class ClipOpenAi():
    def __init__(self, device=None):
        self.device = get_device(device)

    def load_model(self, model_name='ViT-L/14'):
        model, preprocess = clip.load(model_name, self.device)
        self.model = model
        self.preprocess = preprocess

    def unload_model(self):
        del self.model
        del self.processor

    def get_image_features(self, image, needs_grad=False):
        if not needs_grad:
            torch.no_grad()

        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)

        image_features = image_features.to(torch.float32)
        return image_features.to(self.device)

    def preprocess_image_tensor(self, images):
        """
        Preprocesses a given image tensor.

        Args:
        - tensor (torch.Tensor): The input image tensor.

        Returns:
        - torch.Tensor: The preprocessed image tensor.
        """
        # Assert that tensor is 3D and channels = 3
        assert len(images.shape) == 3 and images.shape[0] == 3, "Tensor should be of shape [3, N, M]"

        # Map images to `[0, 1]` space and clip
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)

        # Normalize the image tensor
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(-1, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(-1, 1, 1)
        normalized_image_tensor = (images - mean) / std

        # Resize the image tensor to [N, C, 224, 224] using torch.nn.functional.interpolate
        resized_image_tensor = torch.nn.functional.interpolate(normalized_image_tensor.unsqueeze(0), size=(224, 224),
                                                               mode='bicubic', align_corners=False).squeeze(0)

        return resized_image_tensor

    def get_text_features(self, text, needs_grad=False):
        if needs_grad:
            raise NotImplementedError

        tokens = clip.tokenize(text).to(device=self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens).to(device=self.device, dtype=torch.float32)

        return text_features

