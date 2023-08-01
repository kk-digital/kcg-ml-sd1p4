

import json

class GenerationTask:

    def __init__(self, generation_task_type, prompt, model_name, cfg_strength, iterations, denoiser, seed, output_path):
        self.generation_task_type = generation_task_type
        self.prompt = prompt
        self.model_name = model_name
        self.cfg_strength = cfg_strength
        self.iterations = iterations
        self.denoiser = denoiser
        self.seed = seed
        self.output_path = output_path

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
        }

    def from_dict(self, data):
        return GenerationTask(
        generation_task_type=data['generation_task_type'],
        prompt=data['prompt'],
        model_name=data['model_name'],
        iterations=data['iterations'],
        denoiser=data['denoiser'],
        seed=data['seed'],
        output_path=data['output_path'])

    def save_to_json(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.to_dict(), file)

    def load_from_json(self, filename):
        with open(filename, 'r') as file:
            data = json.load(file)
            return GenerationTask.from_dict(data)