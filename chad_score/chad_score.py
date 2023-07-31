
# using https://github.com/grexzen/SD-Chad/blob/main/simple_inference.py

import torch
import pytorch_lightning as pl
import torch.nn as nn

class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
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

    def forward(self, x):
        return self.layers(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def get_chad_model(model_path):
    model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
    state = torch.load(model_path, map_location=torch.device('cpu') )  # load the model you trained previously or the model available in this repo
    model.load_state_dict(state)
    model.eval()
    return model

def get_chad_score(image_features, model_path="chad_score/models/chad-score-v1.pth"):
    chadModel = get_chad_model(model_path)
    chad_score = chadModel(image_features)
    return chad_score
