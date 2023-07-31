import zipfile
import os
import torch
import clip
import zipfile
import os
import json
import io
import sys
from PIL import Image

sys.path.insert(0, os.getcwd())
from stable_diffusion.model.clip_text_embedder import CLIPTextEmbedder
from stable_diffusion.utils_backend import get_device

def get_image_features(image_data, device):
    model, preprocess = clip.load('ViT-L/14', device)

    # Open the image using Pillow and io.BytesIO
    image = Image.open(io.BytesIO(image_data))

    image_input = preprocess(image).unsqueeze(0).to(device)

    # Encode the image
    with torch.no_grad():
        image_features = model.encode_image(image_input)

    return image_features

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Example usage:
    zip_file_path = 'input/random-shit.zip'

    inputs = []
    expected_outputs = []
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            filename, file_extension = os.path.splitext(file_info.filename)
            if file_extension.lower() == '.jpg':
                json_filename = filename + '.json'
                if json_filename in zip_ref.namelist():
                    jpg_content = zip_ref.read(file_info.filename)
                    json_content = zip_ref.read(json_filename)
                    jpg_filename = file_info.filename
                    json_filename = json_filename

                    image_features = get_image_features(jpg_content, get_device(device))

                    # Decode the bytes to a string
                    json_data_string = json_content.decode('utf-8')
                    # Parse the JSON data into a Python dictionary
                    data_dict = json.loads(json_data_string)
                    # Access properties from the data_dict
                    prompt = data_dict["prompt"]


                    # embed prompt
                    clip_text_embedder = CLIPTextEmbedder(device=get_device(device))
                    clip_text_embedder.load_submodels_auto()

                    embedded_prompts = clip_text_embedder(prompt)

                    print('embedded_prompts')
                    print(embedded_prompts.shape)
                    # Convert the tensor to a flat vector
                    flat_embedded_prompts = torch.flatten(embedded_prompts)

                    with torch.no_grad():
                        flat_vector = flat_embedded_prompts.cpu().numpy()

                    print('flat_vector')
                    print(flat_vector)

                    chad_score = 1.0

                    inputs.append(flat_vector)
                    expected_outputs.append(chad_score)




if __name__ == '__main__':
    main()