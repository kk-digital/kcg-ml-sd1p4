import torch
from torchvision.transforms import ToPILImage

"""
    This script uses this technique (https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204) in order to give a simple preview of the latent space vector    , which is quicker than decoding the image. It also upscales the image to the desired resolution (defaulting to 512x512).
"""

def validate_latent_vector(latent_vector):
    """
    Validate the dimensions of the latent vector.

    Args:
    latent_vector (torch.Tensor): A PyTorch tensor representing the latent vector.

    Raises:
    ValueError: If the latent vector has incorrect dimensions.
    """
    expected_dim = torch.Size([1, 4, 64, 64])
    if images.shape != expected_dim:
        raise ValueError(f"Latent vector should have shape ({expected_dim},), but got {latent_vector.shape}")

def latent_to_pil_image(latent_vector, size=512):
    """
    Convert a latent vector to a PIL image using upscaling.

    Args:
    stable_diffusion (StableDiffusion): The Stable Diffusion object.
    latent_vector (torch.Tensor): A PyTorch tensor representing the latent vector.
    size (int): The size of the output image (default is 512).

    Returns:
    PIL.Image.Image: The generated image.
    """
    # Validate the latent vector
    validate_latent_vector(latent_vector)

    # Decode the latent vector
    decoded_image = to_pil(images.squeeze())

    # Resize the decoded image using upscaling
    resize_transform = torch.nn.functional.interpolate
    resized_image = resize_transform(decoded_image.unsqueeze(0), size=(size, size), mode='bicubic', align_corners=False)

    # Convert the PyTorch tensor to a PIL image
    pil_image = ToPILImage()(resized_image.squeeze())

    return pil_image

def generate_image_from_latent(stable_diffusion, latent_vector):
    """
    Generate an image from a latent vector using Stable Diffusion.

    Args:
    stable_diffusion (StableDiffusion): The Stable Diffusion object.
    latent_vector (torch.Tensor): A PyTorch tensor representing the latent vector.

    Returns:
    PIL.Image.Image: The generated image.
    """
    # Validate the latent vector
    validate_latent_vector(latent_vector)

    # Load the autoencoder and decoder if not already loaded
    stable_diffusion.model.load_autoencoder().load_decoder()

    # Decode the latent vector
    decoded_image = stable_diffusion.decode(latent_vector)

    # Convert the PyTorch tensor to a PIL image
    pil_image = ToPILImage()(decoded_image.squeeze())

    return pil_image
