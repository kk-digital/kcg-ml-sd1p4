import torch
import os

input_dir = os.environ.get('INPUT_DIR')
output_dir = os.environ.get('OUTPUT_DIR')

model = torch.hub.load('username/repo', 'model_name', pretrained=True)

input_data = load_input_data(input_dir)
output_data = model(input_data)
save_output_data(output_data, output_dir)
