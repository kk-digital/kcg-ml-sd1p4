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

    embedded_prompts = []
    for prompt in prompts:
        embeddings = clip_text_embedder.forward(prompt)
        # Flattening tensor and appending
        embedded_prompts.append(embeddings.view(-1))
    embedded_prompts = torch.stack(embedded_prompts)
    #TODO: why is it saving here?
    '''
    torch.save(embedded_prompts, join(EMBEDDED_PROMPTS_DIR, "embedded_prompts.pt"))
    print(
        "Prompts embeddings saved at: ",
        f"{join(EMBEDDED_PROMPTS_DIR, 'embedded_prompts.pt')}",
    )
	'''
	
    ## Clear model from memory
    #get_memory_status()
    clip_text_embedder.to("cpu")
    del clip_text_embedder
    torch.cuda.empty_cache()
    #get_memory_status()

    return embedded_prompts
