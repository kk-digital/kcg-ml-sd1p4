
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
import os
import sys
sys.path.insert(0, os.getcwd())

from typing import List
import torch
from torch import nn, save
from os.path import join
from transformers import CLIPTokenizer, CLIPTextModel
from stable_diffusion2.constants import EMBEDDER_PATH, TOKENIZER_PATH, TRANSFORMER_PATH
from stable_diffusion2.utils.utils import check_device
from torchinfo import summary
# EMBEDDER_PATH = os.path.abspath('./input/model/clip/clip_embedder.ckpt')
# TOKENIZER_PATH = os.path.abspath('./input/model/clip/clip_tokenizer.ckpt')
# TRANSFORMER_PATH = os.path.abspath('./input/model/clip/clip_transformer.ckpt')
class CLIPTextEmbedder(nn.Module):
    """
    ## CLIP Text Embedder
    """

    def __init__(self, version: str = "openai/clip-vit-large-patch14", device=None, max_length: int = 77, tokenizer = None, transformer = None):
        """
        :param version: is the model version
        :param device: is the device
        :param max_length: is the max length of the tokenized prompt
        """
        super().__init__()

        self.device = check_device(device)

        self.version = version

        self.tokenizer = tokenizer
        self.transformer = transformer
        
        self.max_length = max_length
        self.to(self.device)

    def save_submodels(self, tokenizer_path: str = TOKENIZER_PATH, transformer_path: str = TRANSFORMER_PATH):
        torch.save(self.tokenizer, tokenizer_path)
        torch.save(self.transformer, transformer_path)

    def save(self, embedder_path: str = EMBEDDER_PATH):
        torch.save(self, embedder_path)
    
    def load_submodels(self, tokenizer_path = TOKENIZER_PATH, transformer_path = TRANSFORMER_PATH):

        self.tokenizer = torch.load(tokenizer_path, map_location=self.device)
        self.transformer = torch.load(transformer_path, map_location=self.device)
        self.transformer.eval()
        return self
    
    def load_tokenizer(self, tokenizer_path = TOKENIZER_PATH):
        self.tokenizer = torch.load(tokenizer_path, map_location=self.device)
        return self.tokenizer    

    def load_transformer(self, transformer_path = TRANSFORMER_PATH):

        self.transformer = torch.load(transformer_path, map_location=self.device)
        self.transformer.eval()
        return self.transformer


    def unload_submodels(self):
        del self.tokenizer
        del self.transformer
        torch.cuda.empty_cache()
        self.tokenizer = None
        self.transformer = None
    
    def unload_tokenizer(self):
        del self.tokenizer
        torch.cuda.empty_cache()
        self.tokenizer = None

    def unload_transformer(self):
        del self.transformer
        torch.cuda.empty_cache()
        self.transformer = None

    def load_tokenizer_from_lib(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(self.version)
    def load_transformer_from_lib(self):
        self.transformer = CLIPTextModel.from_pretrained(self.version).eval().to(self.device)

    def forward(self, prompts: List[str]):
        """
        :param prompts: are the list of prompts to embed
        """
        # Tokenize the prompts
        batch_encoding = self.tokenizer(prompts, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        # Get token ids
        tokens = batch_encoding["input_ids"].to(self.device)
        
        # Get CLIP embeddings
        return self.transformer(input_ids=tokens).last_hidden_state
#%%
#tests, to be (re)moved

if __name__ == "__main__":
    prompts = ["", "A painting of a computer virus", "A photo of a computer virus"]

    clip = CLIPTextEmbedder()



    clip.load_tokenizer_from_lib()
    clip.load_transformer_from_lib()
    embeddings1 = clip(prompts)

    summary(clip.transformer)
    print("embeddings: ", embeddings1)
    print("embeddings.shape: ", embeddings1.shape)

    clip.save()  

    clip.unload_submodels()

    clip.load_submodels()

    embeddings2 = clip(prompts)

    assert torch.allclose(embeddings1, embeddings2)

    clip = torch.load(EMBEDDER_PATH, map_location="cuda:0")
    print(clip)
    embeddings3 = clip(prompts)
    assert torch.allclose(embeddings1, embeddings3), "embeddings1 != embeddings3"
    assert torch.allclose(embeddings2, embeddings3), "embeddings2 != embeddings3"
    print(os.getcwd())
# %%
