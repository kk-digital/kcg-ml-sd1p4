import torch
import zipfile
import os
import json
import sys
import io
import numpy as np


sys.path.insert(0, os.getcwd())
from generation_task_result import GenerationTaskResult


class ZipDataLoader:
    def load(self, file_path):
        loaded_data = []
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                filename, file_extension = os.path.splitext(file_info.filename)
                if file_extension.lower() == '.jpg':
                    json_filename = filename + '.json'
                    if json_filename in zip_ref.namelist():
                        json_content = zip_ref.read(json_filename)

                        # Decode the bytes to a string
                        json_data_string = json_content.decode('utf-8')

                        # Parse the JSON data into a Python dictionary
                        data_dict = json.loads(json_data_string, cls=NumpyArrayDecoder)

                        image_meta_data = GenerationTaskResult.from_dict(data=data_dict)
                        embedding_name = image_meta_data.embedding_name
                        embedding_content = zip_ref.read(embedding_name)
                        embedding_vector = np.load(io.BytesIO(embedding_content))['data']

                        loaded_data.append({
                            "image_meta_data" : image_meta_data,
                            "embedding_vector" : embedding_vector
                        })

        return loaded_data



# Custom JSON decoder for NumPy arrays
class NumpyArrayDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.json_to_ndarray, *args, **kwargs)

    def json_to_ndarray(self, dct):
        if '__ndarray__' in dct:
            data = np.array(dct['data'], dtype=dct['dtype'])
            if 'shape' in dct:
                data = data.reshape(dct['shape'])
            return data
        return dct

