
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
from torch import nn
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextConfig

import safetensors

from stable_diffusion.constants import TEXT_EMBEDDER_PATH, TOKENIZER_PATH, TEXT_MODEL_PATH
from stable_diffusion.utils.utils import get_device, get_memory_status
from torchinfo import summary
# TEXT_EMBEDDER_PATH = os.path.abspath('./input/model/clip/clip_embedder.ckpt')
# TOKENIZER_PATH = os.path.abspath('./input/model/clip/clip_tokenizer.ckpt')
# TEXT_MODEL_PATH = os.path.abspath('./input/model/clip/clip_transformer.ckpt')
class CLIPTextEmbedder(nn.Module):
    """
    ## CLIP Text Embedder
    """

    def __init__(self, path_tree = None, device=None, max_length: int = 77, tokenizer = None, transformer = None):
        """
        :param version: is the model version
        :param device: is the device
        :param max_length: is the max length of the tokenized prompt
        """
        super().__init__()


        self.path_tree = path_tree
        self.device = get_device(device)

        self.tokenizer = tokenizer
        self.transformer = transformer
        
        self.max_length = max_length
        self.to(self.device)

    def init_submodels(self, tokenizer_path: str = TOKENIZER_PATH, transformer_path: str = TEXT_MODEL_PATH):
        
        config = CLIPTextConfig.from_pretrained(transformer_path, local_files_only=True)
        self.transformer = CLIPTextModel(config).eval().to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        
        return self
        
    def save_submodels(self, tokenizer_path: str = TOKENIZER_PATH, transformer_path: str = TEXT_MODEL_PATH):
        # self.tokenizer.save_pretrained(tokenizer_path, safe_serialization=True)
        # print("tokenizer saved to: ", tokenizer_path)
        self.transformer.save_pretrained(transformer_path, safe_serialization=True)
        # safetensors.torch.save_model(self.transformer, os.path.join(transformer_path, '/model.safetensors'))
        print("transformer saved to: ", transformer_path)
    
    def load_submodels(self, tokenizer_path = TOKENIZER_PATH, transformer_path = TEXT_MODEL_PATH):
        
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        print(f"Tokenizer successfully loaded from : {tokenizer_path}\n")
        self.transformer = CLIPTextModel.from_pretrained(transformer_path, local_files_only=True, use_safetensors = True).eval().to(self.device)  
        # self.init_submodels(tokenizer_path = tokenizer_path, transformer_path = transformer_path)
        # safetensors.torch.load_model(self.transformer, os.path.join(transformer_path, '/model.safetensors'))
        print(f"CLIP text model successfully loaded from : {transformer_path}\n")
        return self

    def unload_submodels(self):

        print("Memory status before unloading submodels: \n")
        get_memory_status()
        if self.tokenizer is not None:
            self.tokenizer
            del self.tokenizer
            self.tokenizer = None
        if self.transformer is not None:
            self.transformer.to("cpu")
            del self.transformer
            self.transformer = None
        torch.cuda.empty_cache()
        print("Memory status after the unloading: \n")
        get_memory_status()        

    def save(self, embedder_path: str = TEXT_EMBEDDER_PATH):
        try:
            safetensors.torch.save_model(self, embedder_path)
            print(f"CLIPTextEmbedder saved to: {embedder_path}")
        except Exception as e:
            print(f"CLIPTextEmbedder not saved. Error: {e}")
            
    def load(self, embedder_path: str = TEXT_EMBEDDER_PATH):
        try:
            safetensors.torch.load_model(self, embedder_path, strict=True)
            print(f"CLIPTextEmbedder loaded from: {embedder_path}")
            return self
        except Exception as e:
            print(f"CLIPTextEmbedder not loaded. Error: {e}")
        
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

if __name__ == "__main__":
    prompts = ["", "A painting of a computer virus", "A photo of a computer virus"]

    clip = CLIPTextEmbedder()



    
    
    embeddings1 = clip(prompts)

    summary(clip.transformer)
    print("embeddings: ", embeddings1)
    print("embeddings.shape: ", embeddings1.shape)

    clip.save()  

    clip.unload_submodels()

    clip.load_submodels()

    embeddings2 = clip(prompts)

    assert torch.allclose(embeddings1, embeddings2)

    clip = torch.load(TEXT_EMBEDDER_PATH, map_location="cuda:0")
    print(clip)
    embeddings3 = clip(prompts)
    assert torch.allclose(embeddings1, embeddings3), "embeddings1 != embeddings3"
    assert torch.allclose(embeddings2, embeddings3), "embeddings2 != embeddings3"
    print(os.getcwd())
# %%
