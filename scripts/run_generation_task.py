import argparse
import sys

base_directory = "./"
sys.path.insert(0, base_directory)

from generation_task import GenerationTask
from text_to_image import text_to_image


def parse_arguments():
    """Command-line arguments for 'classify' command."""
    parser = argparse.ArgumentParser(description="Executes a task file.")

    parser.add_argument('--task_path', type=str, help='Path to the task to execute')

    return parser.parse_args()

def text_to_image_handler(generation_task):
    text_to_image(generation_task.prompt, generation_task.output_path, 'ddim', "./input/model/v1-5-pruned-emaonly.safetensors", True, 50,
                  generation_task.cfg_strength, False, False, 'cuda:0', generation_task.num_images, generation_task.seed)

def main():
    args = parse_arguments()

    task_path = args.task_path

    if task_path is None:
        print('--task_path is a required parameter')
        return

    generation_task = GenerationTask.load_from_json(task_path)

    if generation_task.generation_task_type == 'text_to_image':
        text_to_image_handler(generation_task)


if __name__ == '__main__':
    main()
