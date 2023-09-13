"""
This is a dataset loader function for linear, logistic, and elm regression for generated datasets using Stable Diffusion.
"""

import os
import zipfile
import json
import random
import numpy as np
import time


class GeneratedImageFeatures:
    def __init__(self, prompt: str, model: str, file_name: str, file_hash: str, chad_score_model: str,
                 chad_score: float, seed: int, cfg_strength: int, embedding: [], clip_feature_vector: [],
                 latent_feature: [], prompt_dict: dict, is_score_exist=False, score=0):
        self.prompt = prompt
        self.model = model
        self.file_name = file_name
        self.file_hash = file_hash
        self.chad_score_model = chad_score_model
        self.chad_score = chad_score
        self.seed = seed
        self.cfg_strength = cfg_strength

        self.embedding = embedding
        self.clip_feature_vector = clip_feature_vector
        self.latent_feature = latent_feature
        self.prompt_dict = prompt_dict

        self.is_score_exist = is_score_exist
        self.score = score


class GeneratedImageDataset:
    def __init__(self):
        self.dataset = []

    def load_dataset(self, dataset_path: str):
        start_time = time.time()
        print("Loading dataset...")
        # load zip to ram
        # load zip
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            data_list = []
            file_paths = zip_ref.namelist()
            for file_path in file_paths:
                file_extension = os.path.splitext(file_path)[1]
                if file_extension == ".jpg":
                    # get filename
                    file_path_no_extension = os.path.splitext(file_path)[0]
                    image_dir_path = os.path.dirname(file_path)
                    root_path = os.path.dirname(image_dir_path)
                    features_dir_path = os.path.join(root_path, "features")
                    file_base_name = os.path.basename(file_path_no_extension)

                    # get json
                    file_path_json = file_path_no_extension + ".json"
                    with zip_ref.open(file_path_json) as file:
                        json_content = json.load(file)

                    score_json = ""
                    is_score_exist = False
                    # get score json if exist
                    try:
                        file_path_score_json = file_path_no_extension + ".score.json"
                        with zip_ref.open(file_path_score_json) as file:
                            score_json = json.load(file)

                            # check first if hashes match
                            if json_content["image_hash"] == score_json["image_hash"]:
                                is_score_exist = True
                    except Exception as e:
                        # print("No score json found: {0}".format(e))
                        pass


                    # get embedding
                    file_path_embedding = os.path.join(features_dir_path, file_base_name + ".embedding.npz")
                    with zip_ref.open(file_path_embedding) as file:
                        embedding = np.load(file)
                        embedding_data = embedding['data']

                    # get clip
                    file_path_clip = os.path.join(features_dir_path, file_base_name + ".clip.npz")
                    with zip_ref.open(file_path_clip) as file:
                        clip = np.load(file)
                        clip_data = clip['data']

                    # get latent
                    file_path_latent = os.path.join(features_dir_path, file_base_name + ".latent.npz")
                    with zip_ref.open(file_path_latent) as file:
                        latent = np.load(file)
                        latent_data = latent['data']

                    prompt_dict_data = {}
                    try:
                        # get prompt_dict
                        file_path_prompt_dict = os.path.join(features_dir_path, file_base_name + ".prompt_dict.npz")
                        with zip_ref.open(file_path_prompt_dict) as file:
                            prompt_dict = np.load(file, allow_pickle=True)
                            prompt_dict_data = {"positive_prompt_str": prompt_dict["positive_prompt_str"],
                                                "negative_prompt_str": prompt_dict["negative_prompt_str"],
                                                "num_modifiers": prompt_dict["num_modifiers"],
                                                "num_styles": prompt_dict["num_styles"],
                                                "num_constraints": prompt_dict["num_constraints"],
                                                "prompt_vector": prompt_dict["prompt_vector"].tolist()}
                    except Exception as e:
                        print("No prompt dict found: {0}".format(e))
                    image_features = GeneratedImageFeatures(json_content["prompt"],
                                                            json_content["model"],
                                                            json_content["image_name"],
                                                            json_content["image_hash"],
                                                            json_content["chad_score_model"],
                                                            json_content["chad_score"],
                                                            json_content["seed"],
                                                            json_content["cfg_strength"],
                                                            embedding_data,
                                                            clip_data,
                                                            latent_data,
                                                            prompt_dict_data,
                                                            is_score_exist=is_score_exist)
                    if is_score_exist:
                        image_features.score = score_json['score']

                    # add image features to dataset
                    self.dataset.append(image_features)

        print("Dataset loaded...")
        print("Time elapsed: {0}s".format(format(time.time() - start_time, ".2f")))

    def get_training_and_validation_dataset(self, train_percent=0.5):
        training_dataset = GeneratedImageDataset()
        validation_dataset = GeneratedImageDataset()

        dataset_len = len(self.dataset)
        num_train_data_to_get = int(dataset_len * train_percent)

        # get training dataset
        while len(training_dataset.dataset) < num_train_data_to_get:
            rand_index = random.randint(0, dataset_len - 1)
            data = self.dataset[rand_index]
            if data not in training_dataset.dataset:
                training_dataset.dataset.append(data)

        # get validation dataset
        while len(validation_dataset.dataset) < (dataset_len - num_train_data_to_get):
            rand_index = random.randint(0, dataset_len - 1)
            data = self.dataset[rand_index]
            if (data not in validation_dataset.dataset) and (data not in training_dataset.dataset):
                validation_dataset.dataset.append(data)

        return training_dataset, validation_dataset

    def get_feature_vectors(self):
        feature_vectors = []
        for data in self.dataset:
            feature_vectors.append(data.clip_feature_vector)

        return feature_vectors

    def get_embedding_vector(self):
        embedding_vector = []
        for data in self.dataset:
            embedding_vector.append(data.embedding)

        return embedding_vector

    def get_latent_vector(self):
        latent_vector = []
        for data in self.dataset:
            latent_vector.append(data.latent_feature)

        return latent_vector

    def get_prompt_vectors(self):
        prompt_vectors = []
        for data in self.dataset:
            prompt_vector = data.prompt_dict["prompt_vector"]
            prompt_vectors.append(prompt_vector)

        return prompt_vectors

    def get_chad_scores(self):
        chad_scores = []
        for data in self.dataset:
            chad_scores.append(data.chad_score)

        return chad_scores

    def get_scores(self):
        scores = []

        for data in self.dataset:
            if not data.is_score_exist:
                raise Exception("Scores doesnt exist in dataset")
            scores.append(data.score)

        return scores
