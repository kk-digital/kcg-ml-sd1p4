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

from torch import nn, save
from os.path import join
# from transformers import CLIPTokenizer, CLIPTextModel
import torch
from clip_text_model import CLIPTextModel
from clip_tokenizer import CLIPTokenizer


class CLIPTextEmbedder(nn.Module):
    """
    ## CLIP Text Embedder
    """

    def __init__(self, device="cuda:0", max_length: int = 77):
        """
        :param version: is the model version
        :param device: is the device
        :param max_length: is the max length of the tokenized prompt
        """
        super().__init__()
        self.tokenizer = None
        self.transformer = None
        self.device = device
        self.max_length = max_length

    def load(self, tokenizer_path: str = "./input/model/clip_tokenizer.ckpt", transformer_path: str = "./input/model/clip_transformer.ckpt"):
        self.tokenizer = CLIPTokenizer(device=self.device, max_length=self.max_length)
        self.tokenizer.load(tokenizer_path)
        self.transformer = CLIPTextModel(device=self.device, max_length=self.max_length)
        self.transformer.load(transformer_path)

    def save(self, tokenizer_path: str = "./input/model/clip_tokenizer.ckpt", transformer_path: str = "./input/model/clip_transformer.ckpt", embedder_path: str = "./input/model/clip_embedder.ckpt"):

        torch.save(self, embedder_path) 
        
    def load_from_lib(self, version: str = "openai/clip-vit-large-patch14"):
        self.tokenizer = CLIPTokenizer(device=self.device, max_length=self.max_length)
        self.tokenizer.load_from_lib(version = version)
        self.transformer = CLIPTextModel(device=self.device, max_length=self.max_length)
        self.transformer.load_from_lib(version = version)

    def forward(self, prompts: List[str]):
        """
        :param prompts: are the list of prompts to embed
        """
        # Tokenize the prompts
        tokens = self.tokenizer(prompts, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        # Get CLIP embeddings
        return self.transformer(input_ids = tokens).last_hidden_state

# %%
prompts = ["", "A painting of a computer virus", "A photo of a computer virus"]

clip = CLIPTextEmbedder()

tok = CLIPTokenizer(); tok.load_from_lib()
transformer = CLIPTextModel(); transformer.load_from_lib()
# %%
tokens = tok(prompts)
print(tokens)
# %%
embeddings = transformer(tokens)
# %%

# %%
clip.tokenizer = tok
clip.transformer = transformer
#%%
embeddings2 = clip(prompts)
# %%
embeddings.shape
# %%
embeddings2.shape
# %%
(embeddings == embeddings2).all()

# %%
