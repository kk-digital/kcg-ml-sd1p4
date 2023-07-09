import os.path

import modules.scripts as scripts
import gradio as gr

from modules import sd_samplers, shared
from modules.processing import Processed, process_images, StableDiffusionProcessing, create_infotext
import modules.images as images
from modules.shared import opts, cmd_opts, state

#create chad scorer

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
from PIL import Image
import math
import os
import sys
import traceback

state_name = "sac+logos+ava1-l14-linearMSE.pth"
if not Path(state_name).exists():
    url = f"https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/{state_name}?raw=true"
    import requests
    r = requests.get(url)
    with open(state_name, "wb") as f:
        f.write(r.content)

class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
# load the model you trained previously or the model available in this repo
pt_state = torch.load(state_name, map_location=torch.device('cpu')) 

# CLIP embedding dim is 768 for CLIP ViT L 14
predictor = AestheticPredictor(768)
predictor.load_state_dict(pt_state)
predictor.to(device)
predictor.eval()

clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)

def get_image_features(image, device=device, model=clip_model, preprocess=clip_preprocess):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        # l2 normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.cpu().detach().numpy()
    return image_features

def get_score(image):
    image_features = get_image_features(image)
    score = predictor(torch.from_numpy(image_features).to(device).float())
    return score.item()

#score images

class Script(scripts.Script):

    def title(self):
        return "SD Chad - Stable Diffusion Aesthetic Scorer"

    def ui(self, is_txt2img):
        pass

    def show(self, is_txt2img):
        return True

    def ui(self, is_txt2img):
        n = gr.Textbox(label="Number of images to generate")
        return [n]

    def run(self, p, n):
      for x in range(int(n)):
            #p.prompt = ''
            #print(p.prompt)
            p.seed += 1
            print(p.seed)
            proc = process_images(p)
            gens = proc.images
            #print(gens)
            chad_score = round(get_score(gens[0]),1)
            print(chad_score)
            if chad_score >= 6.9:
                save_chad = images.save_image(gens[0], p.outpath_samples, "", p.seed, "-Chad--" + str(chad_score), opts.samples_format)
                print("Chad")
            else:
                print("Trash")
      return Processed(p, gens, p.seed, "")
