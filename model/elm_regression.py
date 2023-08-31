import os
import torch.nn as nn
import torch
from datetime import datetime

class ELMRegressionModel():
    def __init__(self, device=None):
        self.model_type = 'elm-regression'
        self.date = datetime.now().strftime("%Y-%m-%d")
        self._input_size = None
        self._hidden_layer_neuron_count = None
        self._output_size = None

        self._orthogonalize_weights = False
        self._weight = None
        self._beta = None
        self._bias = None

        if not device and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        self._device = torch.device(device)

        self.activation_func_name = None
        self._activation = None
        self.loss_func_name = "mse"
        self._loss = nn.MSELoss()

    def predict(self, dataset_feature_vector, device=None):
        if device == None:
            device = self._device
        weights = torch.tensor(self._weight, device=device)
        bias = torch.tensor(self._bias, device=device)
        preresult = torch.add(dataset_feature_vector.mm(weights), bias)
        h = self._activation(preresult)
        out = h.mm(self._beta)

        return out

    def compute_gradient(self, input_vector, real_output):
        input_tensor = input_vector.clone().detach().requires_grad_(True)
        # predicted_output = self.model(input_tensor)
        predicted_output = self.predict(input_tensor)

        mse_loss = nn.MSELoss()
        loss = mse_loss(predicted_output, real_output)
        loss.backward()

        input_gradient = input_tensor.grad.data

        input_tensor.to("cpu")
        del input_tensor

        return input_gradient

    @staticmethod
    def get_activation_func(activation_func_name):
        if activation_func_name == "sigmoid":
            return nn.Sigmoid()
        elif activation_func_name == "relu":
            return nn.ReLU()

    def load(self, model_path):
        if not os.path.exists(model_path):
            raise Exception("Model path does not exist")

        checkpoint = torch.load(model_path)

        self._input_size = checkpoint['input_size']
        self._hidden_layer_neuron_count = checkpoint['hidden_layer_neuron_count']
        self._output_size = checkpoint['output_size']

        self._orthogonalize_weights = checkpoint['orthogonalize_weights']
        self._weight = checkpoint['weight']
        self._beta = checkpoint['beta']
        self._bias = checkpoint['bias']

        self.activation_func_name = checkpoint['activation_func']
        self._activation = self.get_activation_func(self.activation_func_name)
