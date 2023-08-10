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

    def compute_gradient(self, input_vector, real_output):
        input_tensor = input_vector.clone().detach().requires_grad_(True)
        predicted_output = self.model(input_tensor)

        self.model.zero_grad()

        mse_loss = nn.MSELoss()
        loss = mse_loss(predicted_output, real_output)
        loss.backward()

        input_gradient = input_tensor.grad.data

        input_tensor.to("cpu")
        del input_tensor

        return input_gradient

    def save(self, model_path):
        # Saving the model to disk
        torch.save(self.model.state_dict(), model_path)

    def load(self, model_path):
        # Loading state dictionary
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
