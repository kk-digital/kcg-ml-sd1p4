# clip_text.py

import torch

from configs.model_config import ModelConfig
from stable_diffusion import CLIPconfigs
from stable_diffusion.model.clip_text_embedder import CLIPTextEmbedder
from stable_diffusion.utils_backend import get_device


# TODO, refactor
# have load
# have unload

# def embed_and_save_prompts(prompts: list, null_prompt=NULL_PROMPT):
# def embed_and_save_prompts(prompts: list):
def clip_text_get_prompt_embedding(config: ModelConfig, prompts: list):
    # null_prompt = null_prompt
    prompts = [prompt.prompt_str for prompt in prompts]

    # load model from memory
    clip_text_embedder = CLIPTextEmbedder(device=get_device())
    clip_text_embedder.load_submodels(**config.get_model(CLIPconfigs.TOKENIZER, CLIPconfigs.TEXT_MODEL, to_dict=True))

    prompt_embedding_list = []
    for prompt in prompts:
        prompt_embedding = clip_text_embedder.forward(prompt)
        prompt_embedding_list.append(prompt_embedding)
        # Flattening tensor and appending
        # print("clip_text_get_prompt_embedding, 1 embedding= ", str(torch.Tensor.size(prompt_embedding)))
        # clip_text_get_prompt_embedding, 1 embedding=  torch.Size([1, 77, 768])
        # prompt_embedding = prompt_embedding.view(-1)
        # print("clip_text_get_prompt_embedding, 2 embedding= ", str(torch.Tensor.size(prompt_embedding)))
        # clip_text_get_prompt_embedding, 2 embedding=  torch.Size([59136])

    prompt_embedding_list = torch.stack(prompt_embedding_list)

    ## Clear model from memory
    clip_text_embedder.to("cpu")
    del clip_text_embedder
    torch.cuda.empty_cache()

    return prompt_embedding_list
