import torch
import gc
import clip
import os
import zipfile
from PIL import Image, UnidentifiedImageError
import sys
import hashlib
import time
from transformers import AutoProcessor, CLIPVisionModel

# no max for image pixel size
Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, os.path.join(os.getcwd(), 'utils', 'dataset'))

from utils.dataset.image_dataset_storage_format.constants import list_of_supported_image_extensions


# TODO: this will all be removed after we have calculation of clip in
#  image dataset storage format cli clip feature zip loader will be deleted
class ClipFeatureZipLoader:
    def __init__(self, verbose=True, clip_skip=False):
        self.verbose = verbose
        self.clip_model = ""
        self.feature_type = "clip"
        self._clip_skip = clip_skip
        if clip_skip:
            self.feature_type += "-skip"

    def load_clip(self, clip_model='ViT-L/14'):
        self.clip_model = clip_model

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and self.verbose:
            print("CUDA is not available. Running on CPU.")

        if self.verbose: print("Loading CLIP " + clip_model)
        if self._clip_skip:
            self.model = CLIPVisionModel.from_pretrained(clip_model).to(self.device)
            self.preprocess = AutoProcessor.from_pretrained(clip_model)
        else:
            self.model, self.preprocess = clip.load(clip_model, device=self.device)
        if self.verbose: print("CLIP loaded succesfully.")

    def unload_clip(self):
        self.model = None
        self.preprocess = None
        gc.collect()
        torch.cuda.empty_cache()
        if self.verbose: print("CLIP unloaded.")

    def compute_feature_vectors(self, opened_images, batch_size):
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

            with torch.no_grad():
                batch_features = self.model.encode_image(batch_inputs)

            image_features.extend(batch_features)

        if self.verbose:
            print("Computed CLIP features for " + str(len(image_features)) + " images.")
            print("Time elapsed: {0}s".format(format(time.time() - start_time, ".2f")))

        return image_features

    def compute_feature_vectors_hidden_layer(self, opened_images):
        start_time = time.time()
        image_features = []
        if self.verbose: print("Computing CLIP features: ")
        for image in opened_images:
            input = self.preprocess(images=image, return_tensors="pt")

            with torch.no_grad():
                output = self.model(**input, output_hidden_states="hidden")
                output = output.hidden_states[11]

            image_features.extend(output)

        if self.verbose:
            print("Computed CLIP features for " + str(len(image_features)) + " images.")
            print("Time elapsed: {0}s".format(format(time.time() - start_time, ".2f")))

        return image_features

    def load_images_from_zip(self, zip_path):
        features = []
        opened_images = []

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:

            file_paths = zip_ref.namelist()
            image_file_paths = [file_path for file_path in file_paths if file_path.lower().endswith(
                ('.gif', '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif', '.tiff', '.webp'))]  # get only image files

            for image_file_path in image_file_paths:
                try:
                    with zip_ref.open(image_file_path) as file:
                        # get hash
                        file_name = os.path.basename(image_file_path)
                        file_hash = (hashlib.sha256(file.read()).hexdigest())
                        file_archive = os.path.basename(zip_path)
                        file_path = "/".join(image_file_path.split("/")[1:])
                        features.append({'file-name': file_name, 'file-hash': file_hash, 'file-path': file_path,
                                         'file-archive': file_archive})

                        opened_images.append(Image.open(file).convert("RGB"))
                except (UnidentifiedImageError, OSError):
                    if self.verbose: print('Skipped image due to error: ' + image_file_path)
                    continue

        if self.verbose: print('Loaded ' + str(len(opened_images)) + ' images inside zip.')

        return features, opened_images

    def load_single_image(self, image_path):
        features = []
        opened_images = []

        try:
            with Image.open(image_path) as image:
                # get hash
                file_name = os.path.basename(image_path)
                file_hash = (hashlib.sha256(image.tobytes()).hexdigest())
                file_archive = os.path.basename(image_path)
                file_path = "/".join(image_path.split("/")[1:])
                features.append({'file-name': file_name, 'file-hash': file_hash, 'file-path': file_path,
                                 'file-archive': file_archive})

                opened_images.append(image.convert("RGB"))
        except (UnidentifiedImageError, OSError):
            if self.verbose: print('Skipped image due to error: ' + image_path)

        if self.verbose: print('Loaded ' + str(len(opened_images)) + ' images inside zip.')

        return features, opened_images

    def get_images_feature_vectors(self, file_path, batch_size=4):
        if file_path.lower().endswith(".zip"):
            features, opened_images = self.load_images_from_zip(file_path)
        elif file_path.lower().endswith(tuple(list_of_supported_image_extensions)):
            features, opened_images = self.load_single_image(file_path)
        else:
            print("Path not supported.")
            return

        if self._clip_skip:
            feature_vectors = self.compute_feature_vectors_hidden_layer(opened_images)
        else:
            feature_vectors = self.compute_feature_vectors(opened_images, batch_size)

        del opened_images  # clean memory

        data_list = []

        for features, feature_vector in zip(features, feature_vectors):
            data = {'file-name': features['file-name'],
                    'file-hash': features['file-hash'],
                    'file-path': features['file-path'],
                    'file-archive': features['file-archive'],
                    'feature-type': self.feature_type,
                    'feature-model': self.clip_model,
                    'feature-vector': feature_vector.tolist()}
            data_list.append(data)

        return data_list
