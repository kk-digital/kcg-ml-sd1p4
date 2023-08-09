import json

import numpy as np


class GenerationTask:

    def __init__(self, generation_task_type, prompt, model_name, cfg_strength, iterations, denoiser, seed, output_path,
                 num_images, image_width, image_height, batch_size, checkpoint_path, flash, device, sampler, steps,
                 force_cpu, num_datasets):
        self.generation_task_type = generation_task_type
        self.prompt = prompt
        self.model_name = model_name
        self.cfg_strength = cfg_strength
        self.iterations = iterations
        self.denoiser = denoiser
        self.seed = seed
        self.output_path = output_path
        self.num_images = num_images
        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.flash = flash
        self.device = device
        self.sampler = sampler
        self.steps = steps
        self.force_cpu = force_cpu
        self.num_datasets = num_datasets

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
            'num_images': self.num_images,
            'image_width': self.image_width,
            'image_height': self.image_height,
            'batch_size': self.batch_size,
            'checkpoint_path': self.checkpoint_path,
            'flash': self.flash,
            'device': self.device,
            'sampler': self.sampler,
            'steps': self.steps,
            'force_cpu': self.force_cpu,
            'num_datasets': self.num_datasets,
        }

    def from_dict(data):
        return GenerationTask(
            generation_task_type=data.get('generation_task_type', ''),
            prompt=data.get('prompt', ''),
            cfg_strength=data.get('cfg_strength', 7),
            model_name=data.get('model_name', ''),
            iterations=data.get('iterations', ''),
            denoiser=data.get('denoiser', ''),
            seed=data.get('seed', ''),
            output_path=data.get('output_path', ''),
            num_images=data.get('num_images', 1),
            image_width=data.get('image_width', 512),
            image_height=data.get('image_height', 512),
            batch_size=data.get('batch_size', 1),
            checkpoint_path=data.get('checkpoint_path', ''),
            flash=data.get('flash', False),
            device=data.get('device', 'cuda'),
            sampler=data.get('sampler', 'ddim'),
            steps=data.get('steps', 50),
            num_datasets=data.get('num_datasets', 1),
            force_cpu=data.get('force_cpu', False))

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
