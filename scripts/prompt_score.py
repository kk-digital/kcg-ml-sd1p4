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
import sys
import random
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


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # One input feature and one output neuron

    def forward(self, x):
        return self.linear(x)

def split_data(input_list, validation_ratio=0.2):
    # Calculate the number of samples for validation and train sets
    num_validation_samples = int(len(input_list) * validation_ratio)
    num_train_samples = len(input_list) - num_validation_samples

    # Shuffle the input_list to ensure random sampling
    random.shuffle(input_list)

    # Split the input_list into validation and train lists
    validation_list = input_list[:num_validation_samples]
    train_list = input_list[num_validation_samples:]

    return validation_list, train_list

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


    linear_regression_model = LinearRegressionModel(77*768)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(linear_regression_model.parameters(), lr=0.001)

    num_inputs = len(inputs)

    validation_inputs, train_inputs = split_data(inputs, validation_ratio=0.4)
    validation_outputs, train_outputs = split_data(expected_outputs, validation_ratio=0.4)

    train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
    train_outputs = torch.tensor(train_outputs, dtype=torch.float32)
    validation_inputs = torch.tensor(validation_inputs, dtype=torch.float32)
    validation_outputs = torch.tensor(validation_outputs, dtype=torch.float32)

    train_outputs = train_outputs.unsqueeze(1)
    validation_outputs = validation_outputs.unsqueeze(1)

    print('train_inputs : shape : ', train_inputs.shape);
    print('train_outputs : shape : ', train_outputs.shape);
    print('validation_inputs : shape : ', validation_inputs.shape);
    print('validation_outputs : shape : ', validation_outputs.shape);

    num_epochs = 1
    for epoch in range(num_epochs):
        # Forward pass
        outputs = linear_regression_model(train_inputs)

        # Compute the loss
        loss = loss_fn(outputs, train_outputs)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Step 6: Evaluate the Model
    with torch.no_grad():
        test_X = validation_inputs
        predicted = linear_regression_model(test_X)

        epsilon = 0.4
        corrects = 0

        predicted = predicted.tolist()
        validation_outputs = validation_outputs.tolist()

        print(test_X.shape)
        print('predicted : ', predicted)
        print('validation_outputs : ', validation_outputs)
        for index, prediction in enumerate(predicted):
            if (abs(predicted[index][0] - validation_outputs[index][0]) < epsilon):
                corrects += 1

    validation_accuracy = corrects / validation_inputs.size(0)

    print('loss : ', loss)
    print('validation_accuracy : ', validation_accuracy)
    print('rate : ', (corrects / num_inputs))
    print('total number of images : ', num_inputs)
    print('total train image count ', train_inputs.size(0))
    print('total validation image count ', validation_inputs.size(0))

if __name__ == '__main__':
    main()