
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
from transformers import CLIPTokenizer, CLIPTextModel
from safetensors.torch import save_model, load_model, load_file

class CLIPTextEmbedder(nn.Module):
    """
    ## CLIP Text Embedder
    """

    def __init__(self, version: str = "openai/clip-vit-large-patch14", device="cuda:0", max_length: int = 77):
        """
        :param version: is the model version
        :param device: is the device
        :param max_length: is the max length of the tokenized prompt
        """
        super().__init__()
        self.version = version
        self.model = None
        self.tokenizer = None
        self.transformer = None
        self.device = device
        self.max_length = max_length
    
    def save(self, tokenizer_path: str = "./clip_tokenizer.ckpt", transformer_path: str = "./clip_transformer.ckpt"):
        torch.save(self.tokenizer, tokenizer_path)
        torch.save(self.transformer, transformer_path)
        # save_model(self.tokenizer, tokenizer_path)
        # save_model(self.transformer, transformer_path)
    
    def load(self, tokenizer_path: str = "./clip_tokenizer.ckpt", transformer_path: str = "./clip_transformer.ckpt"):
        self.tokenizer = torch.load(tokenizer_path, map_location=self.device)
        self.transformer = torch.load(transformer_path, map_location=self.device)        
        # self.tokenizer = load_model(tokenizer_path, map_location=self.device)
        # self.transformer = load_model(transformer_path, map_location=self.device)
    
    def unload(self):        
        del self.tokenizer
        del self.transformer
        torch.cuda.empty_cache()
        self.tokenizer = None
        self.transformer = None

    def hugload(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(self.version)
        # Load the CLIP transformer
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
# %%
prompts = ["", "A painting of a computer virus", "A photo of a computer virus"]

clip = CLIPTextEmbedder()


# %%
clip.hugload()
embbedings1 = clip(prompts)
#%%

clip.save()  
# %%
clip.unload()
# %%
clip.load()
# %%
embeddings2 = clip(prompts)
# %%
torch.allclose(embbedings1, embeddings2)
# %%
