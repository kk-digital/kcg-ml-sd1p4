import torch.nn as nn
import torch

class LinearRegressionModel():
    def __init__(self, input_size, device):
        self.model = nn.Sequential(
            nn.Linear(input_size, 1),
            nn.Identity()
        ).to(device)

        self.model_type = 'linear-regression'
        self.device = device

    def load(self, model_path):
        # Loading state dictionary
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)