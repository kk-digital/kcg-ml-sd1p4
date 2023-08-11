import argparse
import hashlib
import io
import os
import struct
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


sys.path.insert(0, os.getcwd())
from model.linear_regression import LinearRegressionModel
from model.util_data_loader import ZipDataLoader
from model.util_histogram import UtilHistogram


def split_data(input_list, validation_ratio=0.2):
    # Calculate the number of samples for validation and train sets
    num_validation_samples = int(len(input_list) * validation_ratio)

    # Split the input_list into validation and train lists
    validation_list = input_list[:num_validation_samples]
    train_list = input_list[num_validation_samples:]

    return validation_list, train_list

def parse_arguments():
    """Command-line arguments for 'classify' command."""
    parser = argparse.ArgumentParser(description="Training linear model on image promps with chad score.")

    parser.add_argument('--input_path', type=str, default='./input/set_0000.zip', help='Path to input zip')
    parser.add_argument('--output_path', type=str, default='./output/', help='Path output folder')
    parser.add_argument('--model_output_name', type=str, default='prompt_score.pth', help='Filename of the trained model')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs (default: 1000)')
    parser.add_argument('--epsilon_raw', type=float, default=10.0, help='Epsilon for raw data (default: 10.0)')
    parser.add_argument('--epsilon_scaled', type=float, default=0.2, help='Epsilon for scaled data (default: 0.2)')
    parser.add_argument('--use_76th_embedding', action='store_true',
                        help='If this option is set, only use the last entry in the embeddings tensor')
    parser.add_argument('--show_validation_loss', action='store_true', help="whether to show validation loss")

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
    show_validation_loss = args.show_validation_loss
    model_output_name = args.model_output_name
    output_path = args.output_path
    models_dir = os.path.join(output_path, 'models')

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    util_histogram = UtilHistogram()

    inputs = []
    expected_outputs = []

    zip_data_loader = ZipDataLoader()

    loaded_data = []
    if os.path.isdir(zip_file_path):
        file_list = os.listdir(zip_file_path)
        zip_files = [file for file in file_list if file.endswith('.zip')]
        for zip_file_name in zip_files:
            final_path = os.path.join(zip_file_path, zip_file_name)
            loaded_data = loaded_data + zip_data_loader.load(final_path)
    else:
        loaded_data = zip_data_loader.load(zip_file_path)

    for element in loaded_data:
        image_meta_data = element['image_meta_data']
        embedding_vector = element['embedding_vector']

        if use_76th_embedding:
            embedding_vector = embedding_vector[:, 76]

        embedding_vector = torch.tensor(embedding_vector, dtype=torch.float32, device=device);
        # Convert the tensor to a flat vector
        flat_embedded_prompts = torch.flatten(embedding_vector)

        with torch.no_grad():
            flat_vector = flat_embedded_prompts.cpu().numpy()

        # free residual memory
        flat_embedded_prompts.detach()
        del flat_embedded_prompts
        torch.cuda.empty_cache()

        chad_score = image_meta_data.chad_score

        inputs.append(flat_vector)
        expected_outputs.append(chad_score)

    linear_regression_model = LinearRegressionModel(len(inputs[0]), device)
    mse_loss = nn.MSELoss()
    learning_rate = 0.00001
    optimizer = optim.SGD(linear_regression_model.model.parameters(), lr=learning_rate)

    num_inputs = len(inputs)

    validation_inputs, train_inputs = split_data(inputs, validation_ratio=0.2)
    validation_outputs, target_outputs = split_data(expected_outputs, validation_ratio=0.2)

    train_inputs = torch.tensor(train_inputs, dtype=torch.float32, device=device)
    target_outputs = torch.tensor(target_outputs, dtype=torch.float32, device=device)
    validation_inputs = torch.tensor(validation_inputs, dtype=torch.float32, device=device)
    validation_outputs = torch.tensor(validation_outputs, dtype=torch.float32, device=device)

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

    last_validation_lost = 0

    for epoch in range(num_epochs):
        done = False
        optimizer.zero_grad()

        # Forward pass
        model_outputs_raw = linear_regression_model.model(train_inputs)
        model_outputs_scaled = torch.sigmoid(model_outputs_raw)

        # target_outputs_sigmoid = torch.sigmoid(target_outputs)
        # Compute the loss
        loss = mse_loss(model_outputs_raw, target_outputs_raw)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Validation step
        model_validation_outputs_raw = linear_regression_model.model(validation_inputs)
        validation_loss = mse_loss(model_validation_outputs_raw, validation_outputs_raw)

        # the last validation loss for the first epoch is the current validation loss
        if (epoch == 0):
            last_validation_lost = validation_loss

        if last_validation_lost < validation_loss:
            print('Validation loss is starting to drop, abort training')
            num_epochs = epoch + 1
            done = True

        last_validation_lost = validation_loss

        if epoch == num_epochs - 1:
            model_outputs_raw = model_outputs_raw.tolist()
            model_outputs_scaled = model_outputs_scaled.tolist()

            for index, prediction in enumerate(model_outputs_scaled):
                residual = abs(model_outputs_scaled[index][0] - target_outputs_scaled[index][0])
                training_residuals.append(residual.cpu())
                if (residual < epsilon_scaled):
                    training_corrects_scaled += 1

                # free residual memory
                residual.detach()
                del residual;
                torch.cuda.empty_cache()

            for index, prediction in enumerate(model_outputs_raw):
                residual = abs(model_outputs_raw[index][0] - target_outputs_raw[index][0])
                if (residual < epsilon_raw):
                    training_corrects_raw += 1

                # free residual memory
                residual.detach()
                del residual;
                torch.cuda.empty_cache()
            min_training_output_raw = min(model_outputs_raw)
            max_training_output_raw = max(model_outputs_raw)
            min_training_output_scaled = min(model_outputs_scaled)
            max_training_output_scaled = max(model_outputs_scaled)

        # Print progress
        if (epoch + 1) % (num_epochs / 10) == 0:
            if show_validation_loss:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f} | Validation Loss: {validation_loss.item():.4f}')
            else:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        if done:
            break

    # Step 6: Evaluate the Model
    with torch.no_grad():
        test_X = validation_inputs
        predicted_raw = linear_regression_model.model(test_X)
        predicted_scaled = torch.sigmoid(predicted_raw)

        validation_corrects_raw = 0
        validation_corrects_scaled = 0

        predicted_raw = predicted_raw.tolist()
        predicted_scaled = predicted_scaled.tolist()
        validation_outputs_raw = validation_outputs_raw.tolist()
        validation_outputs_scaled = validation_outputs_scaled.tolist()

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

    training_residuals_histogram = util_histogram.report_residuals_histogram(training_residuals, "Training")
    validation_residuals_histogram = util_histogram.report_residuals_histogram(validation_residuals, "Validation")

    # Create a string buffer
    train_report_string_buffer = io.StringIO()

    # Append strings to the buffer
    train_report_string_buffer.write('epoch : ' + str(num_epochs))
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('learning_rate {:.7f}'.format(learning_rate))
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('epsilon_raw {:.4f}'.format(epsilon_raw))
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('epsilon_scaled {:.4f}'.format(epsilon_scaled))
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('use_76th_embedding : ' + str(use_76th_embedding))
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('loss : {:.4f}'.format(loss.item()))
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('training_accuracy (raw) : {:.4f} %'.format(training_accuracy_raw * 100))
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('training_accuracy (scaled) : {:.4f} %'.format(training_accuracy_scaled * 100))
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('validation_accuracy (raw) : {:.4f} %'.format(validation_accuracy_raw * 100))
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('validation_accuracy (scaled) : {:.4f} %'.format(validation_accuracy_scaled * 100))
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('min training output (raw) : {:.4f}'.format(min_training_output_raw[0]))
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('max training output (raw) : {:.4f}'.format(max_training_output_raw[0]))
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('min training output (scaled) : {:.4f}'.format(min_training_output_scaled[0]))
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('max training output (scaled) : {:.4f}'.format(max_training_output_scaled[0]))
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('min predictions (raw) : {:.4f}'.format(min(predicted_raw)[0]))
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('max predictions (raw) : {:.4f}'.format(max(predicted_raw)[0]))
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('min predictions (scaled) : {:.4f}'.format(min(predicted_scaled)[0]))
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('max predictions (scaled) : {:.4f}'.format(max(predicted_scaled)[0]))
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('min training residual : {:.4f}'.format(min(training_residuals)))
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('max training residual : {:.4f}'.format(max(training_residuals)))
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('min validation residual : {:.4f}'.format(min(validation_residuals)))
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('max validation residual : {:.4f}'.format(max(validation_residuals)))
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('total number of images : ' + str(num_inputs))
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('total train image count ' + str(train_inputs.size(0)))
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('total validation image count ' + str(validation_inputs.size(0)))
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write(training_residuals_histogram)
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write('\n')
    train_report_string_buffer.write(validation_residuals_histogram)

    # Get the contents of the buffer as a single string
    train_report_string = train_report_string_buffer.getvalue()

    print(train_report_string)

    reports_path = os.path.join(output_path, 'report.txt')
    with open(reports_path, "w", encoding="utf-8") as file:
        file.write(train_report_string)
    print("Reports saved at {}".format(reports_path))

    model_filepath = os.path.join(models_dir, model_output_name)
    print("Saving model ", model_filepath)
    linear_regression_model.save(model_filepath)


if __name__ == '__main__':
    main()
