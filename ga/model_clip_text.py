#clip_text.py

import torch
from os.path import join

from stable_diffusion.model.clip_text_embedder import CLIPTextEmbedder
from stable_diffusion.utils_backend import get_device, get_memory_status

#TODO, refactor
#have load
#have unload

#def embed_and_save_prompts(prompts: list, null_prompt=NULL_PROMPT):
#def embed_and_save_prompts(prompts: list):
def clip_text_get_prompt_embedding(ModelConfig, prompts: list):
    #null_prompt = null_prompt
    prompts = prompts

    #load model from memory
    clip_text_embedder = CLIPTextEmbedder(device=get_device())
    clip_text_embedder.load_submodels(**ModelConfig.embedder_submodels)

    prompt_embedding_list = []
    for prompt in prompts:
        prompt_embedding = clip_text_embedder.forward(prompt)
        # Flattening tensor and appending

        print("clip_text_get_prompt_embedding, 1 embedding= ", str(prompt_embedding))
        prompt_embedding = prompt_embedding.view(-1)
        print("clip_text_get_prompt_embedding, 2 embedding= ", str(prompt_embedding))
        prompt_embedding_list.append(prompt_embedding)

    prompt_embedding_list = torch.stack(prompt_embedding_list)
    #TODO: why is it saving here?
    '''
    torch.save(prompt_embedding_list, join(prompt_embedding_list_DIR, "prompt_embedding_list.pt"))
    print(
        "Prompts embeddings saved at: ",
        f"{join(prompt_embedding_list_DIR, 'prompt_embedding_list.pt')}",
    )
	'''
	
    ## Clear model from memory
    #get_memory_status()
    clip_text_embedder.to("cpu")
    del clip_text_embedder
    torch.cuda.empty_cache()
    #get_memory_status()

    return prompt_embedding_list
