import json
import numpy as np


class GenerationTask:

    def __init__(self, generation_task_type, prompt, model_name, cfg_strength, iterations, denoiser, seed, output_path, num_images):
        self.generation_task_type = generation_task_type
        self.prompt = prompt
        self.model_name = model_name
        self.cfg_strength = cfg_strength
        self.iterations = iterations
        self.denoiser = denoiser
        self.seed = seed
        self.output_path = output_path
        self.num_images = num_images

    def to_dict(self):
        return {
            'generation_task_type': self.generation_task_type,
            'prompt': self.prompt,
            'model_name': self.model_name,
            'cfg_strength': self.cfg_strength,
            'iterations': self.iterations,
            'denoiser': self.denoiser,
            'seed': self.seed,
            'output_path': self.output_path,
            'num_images' : self.num_images
        }

    def from_dict(data):
        return GenerationTask(
        generation_task_type=data['generation_task_type'],
        prompt=data['prompt'],
        cfg_strength = data['cfg_strength'],
        model_name=data['model_name'],
        iterations=data['iterations'],
        denoiser=data['denoiser'],
        seed=data['seed'],
        output_path=data['output_path'],
        num_images=data['num_images'])

    def save_to_json(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.to_dict(), file, cls=NumpyArrayEncoder)

    def load_from_json(filename):
        with open(filename, 'r') as file:
            data = json.load(file, cls=NumpyArrayDecoder)
            return GenerationTask.from_dict(data)


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

# Custom encoder to handle NumPy arrays
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)