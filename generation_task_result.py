
import json
import numpy as np

class GenerationTaskResult:
    def __init__(self, prompt, model, image_name, embedding_name, clip_name, latent_name, image_hash, chad_score_model, chad_score, seed, cfg_strength):
        self.prompt = prompt
        self.model = model
        self.image_name = image_name
        self.embedding_name = embedding_name
        self.clip_name = clip_name
        self.latent_name = latent_name
        self.image_hash = image_hash
        self.chad_score_model = chad_score_model
        self.chad_score = chad_score
        self.seed = seed
        self.cfg_strength = cfg_strength

    def to_dict(self):
        return {
            "prompt": self.prompt,
            "model" : self.model,
            "image_name": self.image_name,
            "embedding_name": self.embedding_name,
            "clip_name": self.clip_name,
            "latent_name": self.latent_name,
            "image_hash": self.image_hash,
            "chad_score_model": self.chad_score_model,
            "chad_score": self.chad_score,
            "seed": self.seed,
            "cfg_strength": self.cfg_strength
        }


    def from_dict(data):
        return GenerationTaskResult(
        prompt=data['prompt'],
        model=data['model'],
        image_name=data['image_name'],
        embedding_name=data['embedding_name'],
        clip_name=data['clip_name'],
        latent_name=data['latent_name'],
        image_hash=data['image_hash'],
        chad_score_model=data['chad_score_model'],
        chad_score=data['chad_score'],
        seed=data['seed'],
        cfg_strength=data['cfg_strength'])


    def save_to_json(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, cls=NumpyArrayEncoder, indent=2)

    def load_from_json(self, filename):
        with open(filename, 'r') as file:
            data = json.load(file, cls=NumpyArrayDecoder)
            return GenerationTaskResult.from_dict(data)



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
