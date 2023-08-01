
import json;

class GenerationTaskResult:
    def __init__(self, clip_embedding, latent, image_name, image_hash, image_latent, image_clip_vector, chad_score_model, chad_score):
        self.clip_embedding = clip_embedding
        self.latent = latent
        self.image_name = image_name
        self.image_hash = image_hash
        self.image_latent = image_latent
        self.image_clip_vector = image_clip_vector
        self.chad_score_model = chad_score_model
        self.chad_score = chad_score

    def to_dict(self):
        return {
            'clip_embedding': self.clip_embedding,
            'latent': self.latent,
            'image_name': self.image_name,
            'image_hash': self.image_hash,
            'image_latent': self.image_latent,
            'image_clip_vector': self.image_clip_vector,
            'chad_score_model': self.chad_score_model,
            'chad_score': self.chad_score
        }


    def from_dict(self, data):
        return GenerationTaskResult(
        clip_embedding=data['clip_embedding'],
        latent=data['latent'],
        image_name=data['image_name'],
        image_hash=data['image_hash'],
        image_latent=data['image_latent'],
        image_clip_vector=data['image_clip_vector'],
        chad_score_model=data['chad_score_model'],
        chad_score=data['chad_score'])


    def save_to_json(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.to_dict(), file)

    def load_from_json(self, filename):
        with open(filename, 'r') as file:
            data = json.load(file)
            return GenerationTaskResult.from_dict(data)
