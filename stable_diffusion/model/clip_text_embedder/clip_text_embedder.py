
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

from safetensors.torch import save_file, load_file
from stable_diffusion.constants import TEXT_EMBEDDER_PATH, TOKENIZER_PATH, TEXT_MODEL_PATH
from stable_diffusion.constants import TEXT_EMBEDDER_PATH, TOKENIZER_PATH, TEXT_MODEL_PATH
from stable_diffusion.utils.utils import get_device
from torchinfo import summary
# TEXT_EMBEDDER_PATH = os.path.abspath('./input/model/clip/clip_embedder.ckpt')
# TOKENIZER_PATH = os.path.abspath('./input/model/clip/clip_tokenizer.ckpt')
# TEXT_MODEL_PATH = os.path.abspath('./input/model/clip/clip_transformer.ckpt')
class CLIPTextEmbedder(nn.Module):
    """
    ## CLIP Text Embedder
    """

    def __init__(self, path_tree, device=None, max_length: int = 77, tokenizer = None, text_model = None):
        """
        :param version: is the model version
        :param device: is the device
        :param max_length: is the max length of the tokenized prompt
        """
        super().__init__()


        self.path_tree = path_tree
        self.device = get_device(device)

        self.version = version


        self.tokenizer = tokenizer
        self.text_model = text_model
        
        self.max_length = max_length
        self.to(self.device)

    def save_submodels(self, tokenizer_path: str = TOKENIZER_PATH, text_model_path: str = TEXT_MODEL_PATH):
        
        torch.save(self.tokenizer, tokenizer_path)
        print("tokenizer saved to: ", tokenizer_path)
        torch.save(self.text_model, text_model_path)
        print("text_model saved to: ", text_model_path)


    def save(self, embedder_path: str = TEXT_EMBEDDER_PATH, use_safetensors = True):
        if not use_safetensors:
            torch.save(self, embedder_path)
            print(f"CLIPTextEmbedder saved to: {embedder_path}")
        else:
            save_file(self.state_dict(), embedder_path)
            print(f"CLIPTextEmbedder saved to: {embedder_path}")
    
    def load_submodels(self, tokenizer_path = TOKENIZER_PATH, text_model_path = TEXT_MODEL_PATH):

        self.tokenizer = torch.load(tokenizer_path, map_location=self.device)
        self.text_model = torch.load(text_model_path, map_location=self.device)
        self.text_model.eval()
        return self
    
    def load_tokenizer(self, tokenizer_path = TOKENIZER_PATH):
        self.tokenizer = torch.load(tokenizer_path, map_location=self.device)
        return self.tokenizer    

    def load_text_model(self, text_model_path = TEXT_MODEL_PATH):

        self.text_model = torch.load(text_model_path, map_location=self.device)
        self.text_model.eval()
        return self.text_model


    def unload_submodels(self):
        del self.tokenizer
        del self.text_model
        torch.cuda.empty_cache()
        self.tokenizer = None
        self.text_model = None
    
    def unload_tokenizer(self):
        del self.tokenizer
        torch.cuda.empty_cache()
        self.tokenizer = None

    def unload_text_model(self):
        del self.text_model
        torch.cuda.empty_cache()
        self.text_model = None

    def load_tokenizer_from_lib(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(self.path_tree.tokenizer_path, local_files_only=True)
    def load_text_model_from_lib(self):
        self.text_model = CLIPTextModel.from_pretrained(self.path_tree.text_model_path).eval().to(self.device)

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
        return self.text_model(input_ids=tokens).last_hidden_state
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

    clip = torch.load(TEXT_EMBEDDER_PATH, map_location="cuda:0")
    print(clip)
    embeddings3 = clip(prompts)
    assert torch.allclose(embeddings1, embeddings3), "embeddings1 != embeddings3"
    assert torch.allclose(embeddings2, embeddings3), "embeddings2 != embeddings3"
    print(os.getcwd())
# %%
