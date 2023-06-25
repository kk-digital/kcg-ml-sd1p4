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
import torch
from os.path import join
from transformers import CLIPTokenizer as _CLIPTokenizer


class CLIPTokenizer(nn.Module):
    """
    ## CLIP Text Embedder
    """

    def __init__(self, device="cuda:0", max_length: int = 77):
        """
        :param device: is the device
        :param max_length: is the max length of the tokenized prompt
        """
        super().__init__()
        
        self.model = None
        self.device = device
        self.max_length = max_length
    
    def load(self, model_path: str = "./input/model/clip_tokenizer.ckpt"):
        
        self.model = torch.load(model_path).eval().to(self.device)

    def save(self, model_path: str = "./input/model/clip_tokenizer.ckpt"):
        
        torch.save(self.model, model_path)

    def load_from_lib(self, version: str = "openai/clip-vit-large-patch14"):
        # Load the CLIP transformer with transformers library
        self.model = _CLIPTokenizer.from_pretrained(version)

        # return self.transformer        

    def forward(self, prompts: List[str], truncation=True, max_length=None, return_length=True,
                return_overflowing_tokens=False, padding="max_length", return_tensors="pt") -> torch.Tensor:
        """
        :param prompts: are the list of prompts to embed
        """
        if max_length is None:
            max_length = self.max_length

        assert self.model is not None, "Model not loaded"
        # Tokenize the prompts
        batch_encoding = self.model(prompts, truncation=truncation, max_length=max_length, return_length=return_length,
                                        return_overflowing_tokens=return_overflowing_tokens, padding=padding, return_tensors=return_tensors)
        # Get token ids
        tokens = batch_encoding["input_ids"].to(self.device)
        
        return tokens
        # return self.transformer(input_ids=tokens).last_hidden_state
prompts = ["", "A painting of a computer virus", "A photo of a computer virus"]
# if __name__ == "__main__":
#     embedder = CLIPTextEmbedder()
#     prompts = ["", "A painting of a computer virus", "A photo of a computer virus"]
#     embeddings = embedder(prompts)
#     save(embedder, "./input/model/clip_embedder.pt")
#     save(embeddings, './input/prompt_embeddings.pt')
# %%
