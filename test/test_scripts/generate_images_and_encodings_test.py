import sys

base_directory = "./"
sys.path.insert(0, base_directory)
from scripts.generate_images_and_encodings import generate_images


def test_generate_images_and_encodings():
    prompt = 'human, tall, salt, food, table, stake, centered, cartoon'
    generate_images(
        prompt=prompt,
        batch_size=1,
        num_iterations=1,
        device='cuda:0',
    )


if __name__ == "__main__":
    test_generate_images_and_encodings()
