
import torch
import torch.nn as nn
import pytorch_lightning as pl
from stable_diffusion.utils_backend import get_device

class ChadScoreModel(pl.LightningModule):
    def __init__(self, input_size, device = None, xcol='emb', ycol='avg_rating'):
        super().__init__()

        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
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
        self.to(get_device(device))

    def forward(self, x):
        return self.layers(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class ChadScorePredictor:
    def __init__(self, input_size=768, device=None):
        self.device = get_device(device)
        self.model = ChadScoreModel(input_size, self.device)

    def load_model(self, model_path="input/model/chad_score/chad-score-v1.pth"):
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def unload_model(self):
        del self.model

    def get_chad_score(self, image_features):
        return self.model(image_features).item()