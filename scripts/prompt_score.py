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

def report_residuals_histogram(residuals, type):
    max_residual = max(residuals)
    hist_bins = np.linspace(0, max_residual, 11)

    histogram, bin_edges = np.histogram(residuals, bins=hist_bins)
    total_residues = len(residuals)

    histogram_string = f"\n{type} Residuals Histogram:\n"
    histogram_string += f"{'Range':<13} {'Percentage':<12} {'Histogram'}\n"

    max_digits = 12  # Maximum digits for percentage (including decimal point)
    for i, count in enumerate(histogram):
        bin_start, bin_end = bin_edges[i], bin_edges[i + 1]
        percentage = (count / total_residues) * 100
        asterisks = int(percentage / 2)
        percentage_str = f"{percentage:.2f}%"
        histogram_string += f"{bin_start:.2f} - {bin_end:.2f}   {percentage_str:<{max_digits}} {'*' * asterisks}\n"

    return histogram_string

def calculate_pairwise_accuracy(predictions, targets, num_samples=16):
    total_correct = 0
    total_sampled = 0

    # If validation set is 200 items, then loop 200 times
    for _ in range(len(targets)):
        sampled_indices_i = np.random.choice(len(predictions), num_samples, replace=False)
        sampled_indices_j = np.random.choice(len(predictions), num_samples, replace=False)

        for i, j in zip(sampled_indices_i, sampled_indices_j):
            # Skipping the same point
            if j == i:
                continue
            prediction_i = predictions[i]
            prediction_j = predictions[j]
            target_i = targets[i]
            target_j = targets[j]

            if (prediction_i < prediction_j) == (target_i < target_j):
                total_correct += 1
            total_sampled += 1

    return total_correct / total_sampled

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
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs (default: 1000)')
    parser.add_argument('--epsilon_raw', type=float, default=10.0, help='Epsilon for raw data (default: 10.0)')
    parser.add_argument('--epsilon_scaled', type=float, default=0.2, help='Epsilon for scaled data (default: 0.2)')
    parser.add_argument('--use_76th_embedding', action='store_true', help='If this option is set, only use the last entry in the embeddings tensor')

    return parser.parse_args()

def main():
    args = parse_arguments()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Example usage:
    zip_file_path = args.input_path
    use_76th_embedding = args.use_76th_embedding
    num_epochs = args.num_epochs
    epsilon_raw = args.epsilon_raw
    epsilon_scaled = args.epsilon_scaled

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

                    if use_76th_embedding:
                        embedding_vector = embedding_vector[:, 76]

                    embedding_vector = torch.tensor(embedding_vector, dtype=torch.float32);
                    # Convert the tensor to a flat vector
                    flat_embedded_prompts = torch.flatten(embedding_vector)

                    with torch.no_grad():
                       flat_vector = flat_embedded_prompts.cpu().numpy()

                    chad_score = image_meta_data.chad_score

                    inputs.append(flat_vector)
                    expected_outputs.append(chad_score)

    linear_regression_model = LinearRegressionModel(len(inputs[0]))
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

    training_corrects_raw = 0
    training_corrects_scaled = 0
    min_training_output_raw = 0
    max_training_output_raw = 0
    min_training_output_scaled = 0
    max_training_output_scaled = 0
    training_residuals = []
    validation_residuals = []
    pairwise_accuracy = 0

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
            model_outputs_raw = model_outputs_raw.tolist()
            model_outputs_scaled = model_outputs_scaled.tolist()
            for index, prediction in enumerate(model_outputs_scaled):
                residual = abs(model_outputs_scaled[index][0] - target_outputs_scaled[index][0])
                training_residuals.append(residual)
                if (residual < epsilon_scaled):
                    training_corrects_scaled += 1
            for index, prediction in enumerate(model_outputs_raw):
                residual = abs(model_outputs_raw[index][0] - target_outputs_raw[index][0])
                if (residual < epsilon_raw):
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

        pairwise_accuracy = calculate_pairwise_accuracy(predicted_scaled, validation_outputs_scaled)

        print(test_X.shape)
        print('predicted (raw) first 10 elements : ', predicted_raw[:10])
        print('expected output (raw) first 10 elements : ', validation_outputs_raw[:10])

        print('predicted (scaled) first 10 elements : ', predicted_scaled[:10])
        print('expected output (scaled) first 10 elements : ', validation_outputs_scaled[:10])

        for index, prediction in enumerate(predicted_raw):
            residual = abs(predicted_raw[index][0] - validation_outputs_raw[index][0])
            if (residual < epsilon_raw):
                validation_corrects_raw += 1
        for index, prediction in enumerate(predicted_scaled):
            residual = abs(predicted_scaled[index][0] - validation_outputs_scaled[index][0])
            validation_residuals.append(residual)
            if (residual < epsilon_scaled):
                validation_corrects_scaled += 1

    validation_accuracy_raw = validation_corrects_raw / validation_inputs.size(0)
    validation_accuracy_scaled = validation_corrects_scaled / validation_inputs.size(0)
    training_accuracy_raw = training_corrects_raw / train_inputs.size(0)
    training_accuracy_scaled = training_corrects_scaled / train_inputs.size(0)

    training_residuals_histogram = report_residuals_histogram(training_residuals, "Training")
    validation_residuals_histogram = report_residuals_histogram(validation_residuals, "Validation")

    print('loss : {:.4f}'.format(loss.item()))
    print('training_accuracy (raw) : {:.4f} %'.format(training_accuracy_raw * 100))
    print('training_accuracy (scaled) : {:.4f} %'.format(training_accuracy_scaled * 100))
    print('validation_accuracy (raw) : {:.4f} %'.format(validation_accuracy_raw * 100))
    print('validation_accuracy (scaled) : {:.4f} %'.format(validation_accuracy_scaled * 100))
    print('min training output (raw) : {:.4f}'.format(min_training_output_raw[0]))
    print('max training output (raw) : {:.4f}'.format(max_training_output_raw[0]))
    print('min training output (scaled) : {:.4f}'.format(min_training_output_scaled[0]))
    print('max training output (scaled) : {:.4f}'.format(max_training_output_scaled[0]))
    print('min predictions (raw) : {:.4f}'.format(min(predicted_raw)[0]))
    print('max predictions (raw) : {:.4f}'.format(max(predicted_raw)[0]))
    print('min predictions (scaled) : {:.4f}'.format(min(predicted_scaled)[0]))
    print('max predictions (scaled) : {:.4f}'.format(max(predicted_scaled)[0]))
    print('min training residual : {:.4f}'.format(min(training_residuals)))
    print('max training residual : {:.4f}'.format(max(training_residuals)))
    print('min validation residual : {:.4f}'.format(min(validation_residuals)))
    print('max validation residual : {:.4f}'.format(max(validation_residuals)))
    print('total number of images : ', num_inputs)
    print('total train image count ', train_inputs.size(0))
    print('total validation image count ', validation_inputs.size(0))
    print(training_residuals_histogram)
    print(validation_residuals_histogram)

    print('pairwise accuracy : ', pairwise_accuracy)

if __name__ == '__main__':
    main()
