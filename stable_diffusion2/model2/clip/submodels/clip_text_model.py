#%%


"""
---
title: CLIP Text Embedder
summary: >
 CLIP embedder to get prompt embeddings for stable diffusion
---

# CLIP Text Embedder

This is used to get prompt embeddings for [stable diffusion](../index.html).
It uses HuggingFace Transformers CLIP model.
"""

from typing import List
import torch
from torch import nn, save
from os.path import join
from transformers import CLIPTextModel as _CLIPTextModel


class CLIPTextModel(nn.Module):
    """
    ## CLIP Text Model
    """

    def __init__(self, device="cuda:0", max_length: int = 77):
        """
        :param version: is the model version
        :param device: is the device
        :param max_length: is the max length of the tokenized prompt
        """
        super().__init__()
        self.transformer = None
        self.device = device
        self.max_length = max_length


    def load(self, model_path: str = "./input/model/clip_text_model.ckpt"):
        
        self.transformer = torch.load(model_path).eval().to(self.device)

    def save(self, model_path: str = "./input/model/clip_text_model.ckpt"):
        
        torch.save(self.transformer, model_path)

    def load_from_lib(self, version: str = "openai/clip-vit-large-patch14"):
        # Load the CLIP transformer with transformers library
        self.transformer = _CLIPTextModel.from_pretrained(version).eval().to(self.device)

        # return self.transformer

    def forward(self, input_ids: torch.Tensor = None):
        """
        :param tokenized_prompts: a batch of embedded prompts; tensor of shape (batch_size, max_length, 768)
        """
        # Get CLIP embeddings
        return self.transformer(input_ids=input_ids).last_hidden_state

# if __name__ == "__main__":
#     embedder = CLIPTextEmbedder()
#     prompts = ["", "A painting of a computer virus", "A photo of a computer virus"]
#     embeddings = embedder(prompts)
#     save(embedder, "./input/model/clip_embedder.pt")
#     save(embeddings, './input/prompt_embeddings.pt')
prompts = ["", "A painting of a computer virus", "A photo of a computer virus"]
# %%
