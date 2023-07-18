from torch import nn
from stable_diffusion2.utils.utils import check_device

class AestheticPredictor(nn.Module):
    def __init__(self, input_size, device = None):
        super().__init__()
        self.input_size = input_size
        self.device = check_device(device)
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
