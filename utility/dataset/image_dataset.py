import json
import os
import random
import zipfile

from utility.dataset.image_dataset_storage_format.validator import ImageDatasetStorageFormatValidator


class ImageFeatures:
    def __init__(self, file_name: str, file_path: str, file_archive: str, file_hash: str, feature_type: str,
                 feature_model: str, feature_vector: []):
        self.file_name = file_name
        self.file_path = file_path  # path + file name
        self.file_archive = file_archive  # the filename of zip
        self.file_hash = file_hash
        self.feature_type = feature_type  # clip or different
        self.feature_model = feature_model
        self.feature_vector = feature_vector

    def get_tag(self) -> str:
        # get parent dir of filepath, if parent dir is images, then no tag return ""
        parent_dir = os.path.split(os.path.dirname(self.file_path))[1]

        if parent_dir != "images":
            # then it must be its tag
            return parent_dir

        # then it has no tag
        return ""


class ImageDataset:
    def __init__(self):
        self.dataset = []
        self.tag_list = []  # list of tags in the dataset if dataset is tagged

    def load_dataset(self, dataset_path: str, is_tagged=False):
        manifest = []
        features = []

        # check if dataset is valid
        validator = ImageDatasetStorageFormatValidator()
        validator.validate_dataset(dataset_path, is_tagged)

        # load zip to ram
        # get manifest and features
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            file_paths = zip_ref.namelist()
            for file_path in file_paths:
                file_name = os.path.basename(file_path)

                if file_name == "manifest.json":
                    with zip_ref.open(file_path) as file:
                        # load manifest json
                        manifest = json.load(file)

                parent_dir = os.path.split(os.path.dirname(file_path))[1]
                file_extension = os.path.splitext(file_path)[1]
                if parent_dir == "features" and file_extension == ".json":
                    # load feature
                    with zip_ref.open(file_path) as file:
                        features = json.load(file)

                if manifest != [] and features != []:
                    break

        # go through manifest to build image features dataset
        for data in manifest:
            file_name = data["file-name"]
            file_hash = data["file-hash"]
            file_path = data["file-path"]

            # get feature_vector from features
            for item in features:
                if item["file-hash"] == file_hash:
                    file_archive = item["file-archive"]
                    feature_type = item["feature-type"]
                    feature_model = item["feature-model"]
                    feature_vector = item["feature-vector"]
                    break

            image_features = ImageFeatures(file_name, file_path, file_archive, file_hash, feature_type, feature_model,
                                           feature_vector)
            if is_tagged:
                # add tag to tag_list
                tag = image_features.get_tag()
                if tag not in self.tag_list:
                    self.tag_list.append(tag)

            # add image features to dataset
            self.dataset.append(image_features)

    def get_tag_list(self):
        return self.tag_list

    def get_training_and_validation_tagged_dataset(self, tag_string: str, train_percent=0.5):
        training_dataset = ImageDataset()
        validation_dataset = ImageDataset()

        # list of data with tag==tag_string
        tag_string_data_list = []
        for data in self.dataset:
            if data.get_tag() == tag_string:
                tag_string_data_list.append(data)

        dataset_len = len(tag_string_data_list)
        num_train_data_to_get = int(dataset_len * train_percent)

        # get training dataset
        while len(training_dataset.dataset) < num_train_data_to_get:
            rand_index = random.randint(0, dataset_len - 1)
            data = tag_string_data_list[rand_index]
            if (data.get_tag() == tag_string) and (data not in training_dataset.dataset):
                training_dataset.dataset.append(data)

        # get validation dataset
        while len(validation_dataset.dataset) < (dataset_len - num_train_data_to_get):
            rand_index = random.randint(0, dataset_len - 1)
            data = tag_string_data_list[rand_index]
            if (data.get_tag() == tag_string) and (data not in validation_dataset.dataset) and (
                    data not in training_dataset.dataset):
                validation_dataset.dataset.append(data)

        return training_dataset, validation_dataset

    def get_training_and_validation_dataset(self, train_percent=0.5):
        training_dataset = ImageDataset()
        validation_dataset = ImageDataset()

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
            feature_vectors.append(data.feature_vector)

        return feature_vectors

    def get_dataset_size_for_a_tag(self, tag_string):
        tagged_data_size = 0
        for data in self.dataset:
            if data.get_tag() == tag_string:
                tagged_data_size += 1

        return tagged_data_size
