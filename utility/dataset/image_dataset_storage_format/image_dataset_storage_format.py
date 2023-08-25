import zipfile
import os
from .constants import *
import json


class ImageDatasetStorageFormat:
    path_to_zip_file = ""
    zip_ref = ""

    def load_zip_to_memory(self, path_to_zip_file: str):
        self.zip_ref = zipfile.ZipFile(path_to_zip_file, 'r', compression=zipfile.ZIP_DEFLATED)
        self.path_to_zip_file = path_to_zip_file

    def check_dir_exists_in_zip(self, dir_name: str):
        file_paths = self.zip_ref.namelist()

        for file_path in file_paths:
            parent_dir = os.path.split(os.path.dirname(file_path))
            parent_dir_name = parent_dir[1]
            second_parent_dir_name = os.path.split(parent_dir[0])[1]
            if parent_dir_name == dir_name or second_parent_dir_name == dir_name:
                return

        raise Exception("{0}: {1}".format(DIR_DOESNT_EXIST, dir_name))

    def check_file_exist_in_root_in_zip(self, file_name: str):
        file_paths = self.zip_ref.namelist()

        for file_path in file_paths:
            if os.path.basename(file_path) == file_name:
                return

        raise Exception("{0}: {1}".format(FILE_DOESNT_EXIST_IN_ROOT, file_name))


class Manifest:
    def __init__(self, file_name,  file_hash, file_path, file_archive, image_type, image_width, image_height, image_size):
        self.file_name = file_name
        self.file_hash = file_hash
        self.file_path = file_path
        self.file_archive = file_archive
        self.image_type = image_type
        self.image_width = image_width
        self.image_height = image_height
        self.image_size = image_size

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Manifest):
            return {
                "file-name": obj.file_name,
                "file-hash": obj.file_hash,
                "file-path": obj.file_path,
                "file-archive": obj.file_archive,
                "image-type": obj.image_type,
                "image-width": obj.image_width,
                "image-height": obj.image_height,
                "image-size": obj.image_size,
            }
        #Let the base class handle the problem.
        return json.JSONEncoder.default(self, obj)