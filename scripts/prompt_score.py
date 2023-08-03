import zipfile
import os
import torch.nn as nn
import torch.optim as optim
import torch
import clip
import zipfile
import os
import json
import io
import numpy as np
import sys
import random
from PIL import Image
import hashlib
import struct
import argparse

sys.path.insert(0, os.getcwd())
from stable_diffusion.model.clip_text_embedder import CLIPTextEmbedder
from stable_diffusion.utils_backend import get_device
from generation_task_result import GenerationTaskResult


def hash_string_to_float32(input_string):
    """
    Hash a string and represent the hash as a 32-bit floating-point number (Float32).

    Parameters:
        input_string (str): The input string to be hashed.

    Returns:
        np.float32: The 32-bit floating-point number representing the hash.
    """
    try:
        # Convert the string to bytes (required for hashlib)
        input_bytes = input_string.encode('utf-8')

        # Get the hash object for the desired algorithm (SHA-256 used here)
        hash_object = hashlib.sha256()

        # Update the hash object with the input bytes
        hash_object.update(input_bytes)

        # Get the binary representation of the hash value
        hash_bytes = hash_object.digest()

        # Convert the first 4 bytes (32 bits) of the hash into a float32
        float32_hash = struct.unpack('<f', hash_bytes[:4])[0]

        return float32_hash
    except ValueError:
        raise ValueError("Error hashing the input string.")

def get_image_features(image_data, device):
    model, preprocess = clip.load('ViT-L/14', device)

    # Open the image using Pillow and io.BytesIO
    image = Image.open(io.BytesIO(image_data))

    image_input = preprocess(image).unsqueeze(0).to(device)

    # Encode the image
    with torch.no_grad():
        image_features = model.encode_image(image_input)

    return image_features


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, 1),
            nn.Identity()
        )

    def forward(self, x):
        return self.linear(x)

def split_data(input_list, validation_ratio=0.2):
    # Calculate the number of samples for validation and train sets
    num_validation_samples = int(len(input_list) * validation_ratio)
    num_train_samples = len(input_list) - num_validation_samples

    # Split the input_list into validation and train lists
    validation_list = input_list[:num_validation_samples]
    train_list = input_list[num_validation_samples:]

    return validation_list, train_list

def report_residuals_histogram(residuals):
    max_residual = max(residuals)
    hist_bins = np.linspace(0, max_residual, 11)

    histogram, bin_edges = np.histogram(residuals, bins=hist_bins)

    histogram_string = "Residuals Histogram:\n"
    for i, count in enumerate(histogram):
        bin_start, bin_end = bin_edges[i], bin_edges[i + 1]
        histogram_string += f"{bin_start:.2f} - {bin_end:.2f}: {'*' * count}\n"

    return histogram_string

# Custom JSON decoder for NumPy arrays
class NumpyArrayDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.json_to_ndarray, *args, **kwargs)

    def json_to_ndarray(self, dct):
        if '__ndarray__' in dct:
            data = np.array(dct['data'], dtype=dct['dtype'])
            if 'shape' in dct:
                data = data.reshape(dct['shape'])
            return data
        return dct

def parse_arguments():
    """Command-line arguments for 'classify' command."""
    parser = argparse.ArgumentParser(description="Training linear model on image promps with chad score.")

    parser.add_argument('--input_path', type=str, help='Path to input zip')

    return parser.parse_args()

def main():
    args = parse_arguments()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Example usage:
    zip_file_path = args.input_path

    inputs = []
    expected_outputs = []
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            filename, file_extension = os.path.splitext(file_info.filename)
            if file_extension.lower() == '.jpg':
                json_filename = filename + '.json'
                if json_filename in zip_ref.namelist():
                    json_content = zip_ref.read(json_filename)


                    # Decode the bytes to a string
                    json_data_string = json_content.decode('utf-8')

                    # Parse the JSON data into a Python dictionary
                    data_dict = json.loads(json_data_string, cls=NumpyArrayDecoder)

                    image_meta_data = GenerationTaskResult.from_dict(data=data_dict)
                    embedding_name = image_meta_data.embedding_name
                    embedding_content = zip_ref.read(embedding_name)
                    embedding_vector = np.load( io.BytesIO(embedding_content))['data']


                    embedding_vector = torch.tensor(embedding_vector, dtype=torch.float32);
                    # Convert the tensor to a flat vector
                    flat_embedded_prompts = torch.flatten(embedding_vector)

                    with torch.no_grad():
                       flat_vector = flat_embedded_prompts.cpu().numpy()

                    chad_score = image_meta_data.chad_score

                    inputs.append(flat_vector)
                    expected_outputs.append(chad_score)


    linear_regression_model = LinearRegressionModel(77 * 768)
    mse_loss = nn.MSELoss()
    optimizer = optim.SGD(linear_regression_model.parameters(), lr=0.00001)

    num_inputs = len(inputs)

    validation_inputs, train_inputs = split_data(inputs, validation_ratio=0.2)
    validation_outputs, target_outputs = split_data(expected_outputs, validation_ratio=0.2)

    train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
    target_outputs = torch.tensor(target_outputs, dtype=torch.float32)
    validation_inputs = torch.tensor(validation_inputs, dtype=torch.float32)
    validation_outputs = torch.tensor(validation_outputs, dtype=torch.float32)

    target_outputs_raw = target_outputs.unsqueeze(1)
    validation_outputs_raw = validation_outputs.unsqueeze(1)

    target_outputs_scaled = torch.sigmoid(target_outputs_raw)
    validation_outputs_scaled = torch.sigmoid(validation_outputs_raw)

    num_epochs = 2000
    epsilon = 0.2
    training_corrects_raw = 0
    training_corrects_scaled = 0
    min_training_output_raw = 0
    max_training_output_raw = 0
    min_training_output_scaled = 0
    max_training_output_scaled = 0
    validation_residuals = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        model_outputs_raw = linear_regression_model(train_inputs)
        model_outputs_scaled = torch.sigmoid(model_outputs_raw)

        #target_outputs_sigmoid = torch.sigmoid(target_outputs)
        # Compute the loss
        loss = mse_loss(model_outputs_raw, target_outputs_raw)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        if epoch == num_epochs - 1:
            for index, prediction in enumerate(model_outputs_scaled):
                if (abs(model_outputs_scaled[index][0] - target_outputs_scaled[index][0]) < epsilon):
                    training_corrects_scaled += 1
            for index, prediction in enumerate(model_outputs_raw):
                if (abs(model_outputs_raw[index][0] - target_outputs_raw[index][0]) < epsilon):
                    training_corrects_raw += 1
            min_training_output_raw = min(model_outputs_raw)
            max_training_output_raw = max(model_outputs_raw)
            min_training_output_scaled = min(model_outputs_scaled)
            max_training_output_scaled = max(model_outputs_scaled)

        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Step 6: Evaluate the Model
    with torch.no_grad():
        test_X = validation_inputs
        predicted_raw = linear_regression_model(test_X)
        predicted_scaled = torch.sigmoid(predicted_raw)

        validation_corrects_raw = 0
        validation_corrects_scaled = 0

        predicted_raw = predicted_raw.tolist()
        predicted_scaled = predicted_scaled.tolist()
        validation_outputs_raw = validation_outputs_raw.tolist()
        validation_outputs_scaled = validation_outputs_scaled.tolist()

        print(test_X.shape)
        print('predicted (raw) first 10 elements : ', predicted_raw[:10])
        print('expected output (raw) first 10 elements : ', validation_outputs_raw[:10])

        print('predicted (scaled) first 10 elements : ', predicted_scaled[:10])
        print('expected output (scaled) first 10 elements : ', validation_outputs_scaled[:10])

        for index, prediction in enumerate(predicted_raw):
            residual = abs(predicted_raw[index][0] - validation_outputs_raw[index][0])
            if (residual < epsilon):
                validation_corrects_raw += 1
        for index, prediction in enumerate(predicted_scaled):
            residual = abs(predicted_scaled[index][0] - validation_outputs_scaled[index][0])
            validation_residuals.append(residual)
            if (residual < epsilon):
                validation_corrects_scaled += 1

    validation_accuracy_raw = validation_corrects_raw / validation_inputs.size(0)
    validation_accuracy_scaled = validation_corrects_scaled / validation_inputs.size(0)
    training_accuracy_raw = training_corrects_raw / train_inputs.size(0)
    training_accuracy_scaled = training_corrects_scaled / train_inputs.size(0)

    residuals_histogram = report_residuals_histogram(validation_residuals)

    print('loss : ', loss.item())
    print('training_accuracy (raw) : ', (training_accuracy_raw * 100), '%')
    print('training_accuracy (scaled) : ', (training_accuracy_scaled * 100), '%')
    print('validation_accuracy (raw) : ', (validation_accuracy_raw * 100), '%')
    print('validation_accuracy (scaled) : ', (validation_accuracy_scaled * 100), '%')
    print('min training output (raw) : ', min_training_output_raw.item())
    print('max training output (raw) : ', max_training_output_raw.item())
    print('min training output (scaled) : ', min_training_output_scaled.item())
    print('max training output (scaled) : ', max_training_output_scaled.item())
    print('min predictions (raw) : ', min(predicted_raw)[0])
    print('max predictions (raw) : ', max(predicted_raw)[0])
    print('min predictions (scaled) : ', min(predicted_scaled)[0])
    print('max predictions (scaled) : ', max(predicted_scaled)[0])
    print('total number of images : ', num_inputs)
    print('total train image count ', train_inputs.size(0))
    print('total validation image count ', validation_inputs.size(0))
    print('\n', residuals_histogram)

if __name__ == '__main__':
    main()
