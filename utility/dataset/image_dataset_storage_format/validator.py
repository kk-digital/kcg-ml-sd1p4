"""
Validator checks zip format and raise exception for what is wrong and missing.
"""

from .image_dataset_storage_format import *


class ImageDatasetStorageFormatValidator(ImageDatasetStorageFormat):
    def validate_dataset(self, path_to_zip_file: str, is_tagged=False, is_generated_dataset=False):
        self.load_zip_to_memory(path_to_zip_file)
        self.check_images_folder(is_tagged, is_generated_dataset)
        self.check_features_folder()
        self.check_file_exist_in_root_in_zip("manifest.json")
        self.check_if_keys_exists_in_json("manifest.json", manifest_json_keys_to_check)
        print("Dataset Validation Complete: {0}".format(path_to_zip_file))

    def check_images_folder(self, is_tagged=False, is_generated_dataset=False):
        print("Checking images folder...")
        self.check_dir_exists_in_zip("images")
        if is_tagged:
            self.__check_all_images_are_in_images_dir_tagged()
        else:
            self.__check_all_images_are_in_images_dir_untagged()

        self.__check_all_files_in_images_dir_are_supported(is_generated_dataset)

    def __check_all_images_are_in_images_dir_untagged(self):
        file_paths = self.zip_ref.namelist()
        for file_path in file_paths:
            name = os.path.basename(file_path)
            file_extension = os.path.splitext(name)[1]
            if file_extension in list_of_supported_image_extensions:
                parent_dir = os.path.split(os.path.dirname(file_path))[1]
                if parent_dir != "images":
                    raise Exception("{0}: {1}".format(IMAGE_NOT_IN_IMAGE_DIR, file_path))

    def __check_all_images_are_in_images_dir_tagged(self):
        file_paths = self.zip_ref.namelist()
        for file_path in file_paths:
            name = os.path.basename(file_path)
            file_extension = os.path.splitext(name)[1]
            if file_extension in list_of_supported_image_extensions:
                parent_dir = os.path.split(os.path.dirname(file_path))
                second_parent_dir_name = os.path.split(parent_dir[0])[1]

                # if tagged, the second parent dir must be images
                if second_parent_dir_name != "images":
                    raise Exception("{0}: {1}".format(IMAGE_NOT_IN_IMAGE_DIR, file_path))

    def __check_all_files_in_images_dir_are_supported(self, is_generated_dataset=False):
        file_paths = self.zip_ref.namelist()
        for file_path in file_paths:
            parent_dir = os.path.split(os.path.dirname(file_path))
            second_parent_dir_name = os.path.split(parent_dir[0])[1]

            if os.path.split(os.path.dirname(file_path))[1] == "images" or second_parent_dir_name == "images":
                name = os.path.basename(file_path)

                if not name == "":
                    file_extension = os.path.splitext(name)[1]
                    if file_extension not in list_of_supported_image_extensions :
                        if (is_generated_dataset is False) or (is_generated_dataset is True and file_extension != ".json"):
                            raise Exception("{0}: {1}".format(IMAGE_EXTENSION_NOT_SUPPORTED, file_path))

    def check_features_folder(self):
        print("Checking features folder...")
        self.check_dir_exists_in_zip("features")
        features_paths = self.__check_all_features_json_are_in_features_dir()
        for path in features_paths:
            self.check_if_keys_exists_in_json(os.path.basename(path), features_json_keys_to_check)

    def __check_all_features_json_are_in_features_dir(self):
        features_paths = []
        file_paths = self.zip_ref.namelist()
        for file_path in file_paths:
            name = os.path.basename(file_path)
            file_extension = os.path.splitext(name)[1]

            # name[0]!="." is added to avoid garbage files created by OS
            # mac creates ._filename inside zip_ref
            if "clip" in name and file_extension == ".json" and name != "manifest.json" and name[0] != ".":
                parent_dir = os.path.split(os.path.dirname(file_path))[1]
                if parent_dir != "features":
                    print(file_path)
                    raise Exception("{0}: {1}".format(FEATURES_JSON_NOT_IN_FEATURES_DIR, file_path))
                features_paths.append(file_path)

        return features_paths


    def check_if_keys_exists_in_json(self, json_file_name: str, keys_to_check: []):
        file_paths = self.zip_ref.namelist()
        for file_path in file_paths:
            name = os.path.basename(file_path)
            if name == json_file_name:
                with self.zip_ref.open(file_path) as file:
                    json_data = json.load(file)
                    if len(json_data) > 0:
                        for item in json_data:
                            for key in keys_to_check:
                                if (item.get(key) is None):
                                    raise Exception("{0}: {1}", KEY_DOESNT_EXIST_IN_JSON, key)





