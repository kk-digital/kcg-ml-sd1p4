import zipfile
import os
import torch
import clip
import zipfile
import os
import io
from PIL import Image

def extract_file_pairs_with_content_from_zip(zip_filename):
    file_pairs = {}

    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            filename, file_extension = os.path.splitext(file_info.filename)
            if file_extension.lower() == '.jpg':
                json_filename = filename + '.json'
                if json_filename in zip_ref.namelist():
                    jpg_content = zip_ref.read(file_info.filename)
                    json_content = zip_ref.read(json_filename)
                    file_pairs[filename] = {
                        'jpg_filename': file_info.filename,
                        'jpg_content': jpg_content,
                        'json_filename': json_filename,
                        'json_content': json_content,
                    }

    return file_pairs





def main():
    # Example usage:
    zip_file_path = 'input/random-shit.zip'

    image = []
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

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model, preprocess = clip.load('ViT-L/14', device)

                    # Open the image using Pillow and io.BytesIO
                    with io.BytesIO(jpg_content) as stream:
                        image = Image.open(stream)
                    image_input = preprocess(image).unsqueeze(0).to(device)

                    # Encode the image
                    with torch.no_grad():
                        image_features = model.encode_image(image_input)

                    chad_score = 1.0

                    print(image_features.tolist())
                    inputs.append(image_features.tolist())
                    expected_outputs.append(chad_score)

                    image.append({
                        'jpg_filename' : jpg_filename,
                        'json_filename' : json_filename,
                        'json_content' : 'cat',
                        'image_features' :
                    })



if __name__ == '__main__':
    main()