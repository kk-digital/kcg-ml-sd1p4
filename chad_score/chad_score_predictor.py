import torch
from torch import nn
from stable_diffusion.utils.utils import get_device


class ChadPredictorModel(nn.Module):
    def __init__(self, input_size=768, device=None):
        super().__init__()
        self.input_size = input_size
        self.device = get_device(device)
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.to(self.device)

    def forward(self, x):
        return self.layers(x)


class ChadPredictor():
    def __init__(self, input_size=768, device=None):
        self.input_size = input_size
        self.device = device
        self.model = None

    def load(self, model_path):
        self.model = ChadPredictorModel(input_size=self.input_size, device=self.device)

        state = torch.load(model_path,
                           map_location=self.device)  # load the model you trained previously or the model available in this repo
        self.model.load_state_dict(state)
        self.model.eval()

    def unload(self):
        del self.model
        torch.cuda.empty_cache()
