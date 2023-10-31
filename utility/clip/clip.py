import torch
import gc
import time
from typing import Optional

from transformers import CLIPModel, CLIPImageProcessor, AutoTokenizer


class ClipModel:
    def __init__(self, verbose=True, clip_skip=False, device=None):
        self.verbose = verbose
        self.clip_model = ""
        self.feature_type = "clip"
        self._clip_skip = clip_skip
        if clip_skip:
            self.feature_type += "-skip"

        self.device = device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu" and self.verbose:
                print("CUDA is not available. Running on CPU.")

        self.model = None
        self.preprocess = None
        self.tokenizer = None

    def load_clip(self, clip_model="openai/clip-vit-large-patch14"):
        if clip_model == "ViT-L/14":
            # this is the name of model when using huggingface transformers
            clip_model = 'openai/clip-vit-large-patch14'
        self.clip_model = clip_model

        if self.verbose: print("Loading CLIP " + clip_model)
        # TODO: remove hard code of paths
        self.model = CLIPModel.from_pretrained("./input/model/clip/vit-large-patch14/vit-large-patch14.safetensors", config="./input/model/clip/vit-large-patch14/config.json")
        self.model = self.model.to(self.device)
        self.preprocess = CLIPImageProcessor.from_pretrained("./input/model/clip/img_enc_processor")
        if self.verbose: print("CLIP loaded succesfully.")

    def load_tokenizer(self):
        print('Loading Clip Tokenizer')
        self.tokenizer = AutoTokenizer.from_pretrained("./input/model/clip/txt_emb_tokenizer/")
        print('Tokenizer loaded successfully')

    def unload_clip(self):
        self.model = None
        self.preprocess = None
        gc.collect()
        torch.cuda.empty_cache()
        if self.verbose: print("CLIP unloaded.")

    def get_image_features(self, image):
        if self.device == "cpu":
            print("CUDA is not available. Running on CPU.")
        inputs = self.preprocess(images=image, return_tensors="pt")

        with torch.no_grad():
            if self._clip_skip:
                # ref https://github.com/huggingface/transformers/blob/41aef33758ae166291d72bc381477f2db84159cf/src/transformers/models/clip/modeling_clip.py#L1086
                vision_outputs = self.model.vision_model(
                    **inputs,
                    output_hidden_states=True,
                )

                # clip-vit-l-14 have 24 layers, we only do until 23
                penultimate_layer_output = vision_outputs.hidden_states[23]

                # ref: https://github.com/huggingface/transformers/blob/41aef33758ae166291d72bc381477f2db84159cf/src/transformers/models/clip/modeling_clip.py#L893C11-L893C11
                pooled_output = penultimate_layer_output[:, 0, :]
                pooled_output = self.model.vision_model.post_layernorm(pooled_output)
                image_features = self.model.visual_projection(pooled_output)

                image_features = image_features.to(torch.float32)
            else:
                image_features = self.model.get_image_features(**inputs, output_hidden_states=True)

        # returns image features and the penultimate layer
        # return image_features.to(self.device), pooled_output.to(self.device)
        return image_features.to(self.device)

    def get_text_features(self, text):
        if self.device == "cpu":
            print("CUDA is not available. Running on CPU.")

        inputs = self.tokenizer(text, padding=True, return_tensors="pt")

        text_features = self.model.get_text_features(**inputs)

        return text_features

    def compute_feature_vectors(self, opened_images, batch_size):
        start_time = time.time()

        num_images = len(opened_images)
        num_batches = (num_images + batch_size - 1) // batch_size

        image_features = []
        penultimate_layer_features = []

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_images)
            batch_images = opened_images[start_idx:end_idx]

            if self.verbose: print("Computing CLIP features for batch of size: " + str(len(batch_images)))

            batch_inputs = self.preprocess(images=batch_images, return_tensors="pt")

            if self._clip_skip:
                # ref https://github.com/huggingface/transformers/blob/41aef33758ae166291d72bc381477f2db84159cf/src/transformers/models/clip/modeling_clip.py#L1086
                batch_vision_outputs = self.model.vision_model(
                    **batch_inputs,
                    output_hidden_states=True,
                )

                # clip-vit-l-14 have 24 layers, we only do until 23
                batch_penultimate_layer_output = batch_vision_outputs.hidden_states[23]

                # ref: https://github.com/huggingface/transformers/blob/41aef33758ae166291d72bc381477f2db84159cf/src/transformers/models/clip/modeling_clip.py#L893C11-L893C11
                batch_pooled_output = batch_penultimate_layer_output[:, 0, :]
                batch_pooled_output = self.model.vision_model.post_layernorm(batch_pooled_output)
                batch_image_features = self.model.visual_projection(batch_pooled_output)

                batch_image_features = batch_image_features.to(torch.float32)

                penultimate_layer_features.extend(batch_pooled_output.to(torch.float32))

            else:
                batch_image_features = self.model.get_image_features(**batch_inputs)

            image_features.extend(batch_image_features)

        if self.verbose:
            print("Computed CLIP features for " + str(len(image_features)) + " images.")
            print("Time elapsed: {0}s".format(format(time.time() - start_time, ".2f")))

        return image_features, penultimate_layer_features

    def compute_feature_vectors_custom_preprocessor(self, opened_images, batch_size):
        start_time = time.time()

        num_images = len(opened_images)
        num_batches = (num_images + batch_size - 1) // batch_size

        image_features = []

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_images)
            batch_images = opened_images[start_idx:end_idx]

            if self.verbose: print("Computing CLIP features for batch of size: " + str(len(batch_images)))

            batch_inputs = torch.stack([self.preprocess(image) for image in batch_images]).to(self.device)
            batch_image_features = self.model.get_image_features(batch_inputs)

            image_features.extend(batch_image_features)

        if self.verbose:
            print("Computed CLIP features for " + str(len(image_features)) + " images.")
            print("Time elapsed: {0}s".format(format(time.time() - start_time, ".2f")))

        return image_features

