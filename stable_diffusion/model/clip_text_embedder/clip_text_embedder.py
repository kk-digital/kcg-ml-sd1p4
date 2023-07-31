# %%


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

import torch
import safetensors as st
from torch import nn
from typing import List
from torchinfo import summary
from transformers import CLIPTokenizer, CLIPTextModel
from stable_diffusion.constants import TEXT_EMBEDDER_PATH, TOKENIZER_PATH, TEXT_MODEL_PATH
from stable_diffusion.utils_backend import get_device, get_memory_status


# TEXT_EMBEDDER_PATH = os.path.abspath('./input/model/clip/clip_embedder.ckpt')
# TOKENIZER_PATH = os.path.abspath('./input/model/clip/clip_tokenizer.ckpt')
# TEXT_MODEL_PATH = os.path.abspath('./input/model/clip/clip_transformer.ckpt')
class CLIPTextEmbedder(nn.Module):
    """
    ## CLIP Text Embedder
    """

    def __init__(self, path_tree=None, device=None, max_length: int = 77, tokenizer=None, text_model=None):
        """
        :param version: is the model version
        :param device: is the device
        :param max_length: is the max length of the tokenized prompt
        """
        super().__init__()

        self.path_tree = path_tree
        self.device = get_device(device)

        self.tokenizer = tokenizer
        self.text_model = text_model

        self.max_length = max_length
        self.to(self.device)

    def save_submodels(self, tokenizer_path: str = TOKENIZER_PATH, text_model_path: str = TEXT_MODEL_PATH):
        self.tokenizer.save_pretrained(tokenizer_path, safe_serialization=True)
        print("tokenizer saved to: ", tokenizer_path)
        self.text_model.save_pretrained(text_model_path, safe_serialization=True)
        print("text_model saved to: ", text_model_path)

    def load_submodels(self, tokenizer_path=TOKENIZER_PATH, text_model_path=TEXT_MODEL_PATH):

        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        print(f"Tokenizer successfully loaded from : {tokenizer_path}\n")
        self.text_model = CLIPTextModel.from_pretrained(text_model_path, local_files_only=True,
                                                        use_safetensors=True).eval().to(self.device)
        print(f"CLIP text model successfully loaded from : {text_model_path}\n")
        return self


    def load_submodels_auto(self):

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(self.device)
        return self
    def unload_submodels(self):

        print("Memory status before unloading submodels: \n")
        get_memory_status()
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.text_model is not None:
            del self.text_model
            self.text_model = None
        torch.cuda.empty_cache()
        print("Memory status after the unloading: \n")
        get_memory_status()

    def save(self, text_embedder_path: str = TEXT_EMBEDDER_PATH, use_safetensors=True):

        st.torch.save_model(self, text_embedder_path, use_safetensors=use_safetensors)
        print(f"CLIPTextEmbedder saved to: {text_embedder_path}")

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
