

from typing import Path, Union
import torch
import safetensors

from stable_diffusion.model.clip_embedder import CLIPTextEmbedder

class CLIPTextEmbedderModel:
    
    def __init__(self, safetensors: bool = True):
        self.safetensors = safetensors
        self.model = None

    def load(self, model: Union(Path, CLIPTextEmbedder)):        
        
        if isinstance(model, Path):
            if not self.safetensors:
                self.model = torch.load(model)
        else:
            self.model = model
    
    def save(self, model: Union(Path, CLIPTextEmbedder)):
        
        if isinstance(model, Path):
            if not self.safetensors:
                torch.save(self.model, model)
        else:
            self.model = model