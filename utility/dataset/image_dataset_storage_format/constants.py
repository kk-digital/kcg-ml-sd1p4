# Exception constants
DIR_DOESNT_EXIST = Exception("Dir doesn't exist in root")
IMAGE_NOT_IN_IMAGE_DIR = Exception("Image not in image dir")
IMAGE_EXTENSION_NOT_SUPPORTED = Exception("Image extension not supported")
FEATURES_JSON_NOT_IN_FEATURES_DIR = Exception("Features json not in features dir")
FILE_DOESNT_EXIST_IN_ROOT = Exception("File doesn't exist in root")
KEY_DOESNT_EXIST_IN_JSON = Exception("Key doesn't exist in json")
DATALIST_IS_EMPTY = Exception("Data list is empty")

# constants
list_of_supported_image_extensions = [".jpg", ".png", ".gif", ".jpeg", '.webp']
manifest_json_keys_to_check = ["file-name", "file-hash", "file-path", "file-archive", "image-type", "image-width", "image-height", "image-size"]
features_json_keys_to_check = ["file-name", "file-hash", "file-path", "file-archive", "feature-type", "feature-model", "feature-vector"]
list_of_expected_folders = ["images", "features"]