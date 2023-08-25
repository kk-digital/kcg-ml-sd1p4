"""
Processor processes all lacking data of an image dataset. It will move all images to /images, generate manifest.json, and generate features json.
"""
import hashlib
import re
import io
from PIL import UnidentifiedImageError
from PIL import Image
from .image_dataset_storage_format import *
from utils.clip.clip_feature_zip_loader import ClipFeatureZipLoader

# no max for image pixel size
Image.MAX_IMAGE_PIXELS = None


class ImageDatasetStorageFormatProcessor(ImageDatasetStorageFormat):
    def format_and_compute_manifest(self, path_to_zip_file: str, is_tagged=False, is_generated_dataset=False,
                                    output_path="./output"):
        self.load_zip_to_memory(path_to_zip_file)
        data_list = self.get_all_supported_files_in_zip(is_tagged, is_generated_dataset)
        data_list = self.compute_manifest(data_list)
        self.save_data_to_zip(data_list, output_path)

    def compute_features_of_zip(self, path_to_zip_file: str, clip_model="ViT-L/14", batch_size=8):
        # remove all special char, replace with dash and use lower case
        json_name = re.sub("[^0-9a-zA-Z]+", "-", clip_model).lower() + ".json"

        # compute features using clip tools
        loader = ClipFeatureZipLoader()
        loader.load_clip(clip_model)

        # TODO: remove hard coding of 'clip' to filename
        #  when we implement getting of features using other
        #  feature type other than 'clip'
        json_name = "clip-" + json_name
        feature_vectors = loader.get_images_feature_vectors(path_to_zip_file, batch_size)

        # save features to features dir in zip
        zip_name = os.path.splitext(os.path.basename(path_to_zip_file))[0]
        save_file_path = os.path.join(zip_name, "features", json_name)
        with zipfile.ZipFile(path_to_zip_file, mode="a", compression=zipfile.ZIP_DEFLATED) as zip_file:
            dumped_json = json.dumps(feature_vectors, indent=4)
            zip_file.writestr(save_file_path, data=dumped_json)

    def get_all_supported_files_in_zip(self, is_tagged=False, is_generated_dataset=False) -> []:
        data_list = []

        file_paths = self.zip_ref.namelist()
        for file_path in file_paths:
            name = os.path.basename(file_path)

            file_extension = os.path.splitext(name)[1]
            if (file_extension in list_of_supported_image_extensions) or (
                    is_generated_dataset is True and file_extension in [".json", ".npz"]):
                parent_path = os.path.split(os.path.dirname(file_path))
                parent_dir_name = parent_path[1]

                # get image data
                image_bytes = self.zip_ref.read(file_path)

                # if tagged, the second parent dir must be images
                if file_extension in list_of_supported_image_extensions or (
                        is_generated_dataset is True and file_extension == ".json"):
                    if is_tagged:
                        file_full_path = os.path.join("images", parent_dir_name, name)
                    else:
                        file_full_path = os.path.join("images", name)
                else:
                    file_full_path = os.path.join("features", name)

                # add proper file path+file name and image to data list
                data_list.append({"file-path": file_full_path, "data": image_bytes})

        return data_list

    def compute_manifest(self, data_list: []) -> []:
        image_manifest_array = []
        for item in data_list:
            file_path = item["file-path"]
            name = os.path.basename(file_path)
            file_extension = os.path.splitext(name)[1]
            if file_extension in list_of_supported_image_extensions:
                image_data = item["data"]

                try:
                    image = Image.open(io.BytesIO(image_data))
                except (UnidentifiedImageError, OSError):
                    print('Skipped empty image: ' + file_path)
                    continue

                image_format = file_extension[1:]
                if image_format == "jpg":
                    image_format = "JPEG"

                file_archive = os.path.basename(self.path_to_zip_file)
                file_name = name
                image_type = file_extension
                image_size = len(image_data)  # use the original image data size
                file_hash = hashlib.sha256(image_data).hexdigest()  # hash the original image data
                image_width, image_height = image.size

                manifest = Manifest(file_name, file_hash, file_path, file_archive, image_type, image_width,
                                    image_height, image_size)
                image_manifest_array.append(manifest)

        # add manifest to data_list
        manifest_path = "manifest.json"
        data_list.append(
            {"file-path": manifest_path, "data": json.dumps(image_manifest_array, cls=CustomEncoder, indent=4)})
        return data_list

    def save_data_to_zip(self, data_list: [], output_path="./output"):
        zip_full_name = os.path.basename(self.path_to_zip_file)
        zip_name = os.path.splitext(zip_full_name)[0]
        output_zip_path = os.path.join(output_path, zip_full_name)

        # if data list is empty, raise exception
        if data_list is None:
            raise DATALIST_IS_EMPTY

        # if output path doesn't exist, create one
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # save processed data
        with zipfile.ZipFile(output_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zip_ref:
            for data in data_list:
                zip_ref.writestr(os.path.join(zip_name, data["file-path"]), data["data"])
        print("Dataset Processing Complete: {0}".format(output_path))
