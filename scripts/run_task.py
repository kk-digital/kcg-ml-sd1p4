



import argparse
from generation_task import GenerationTask

def parse_arguments():
    """Command-line arguments for 'classify' command."""
    parser = argparse.ArgumentParser(description="Executes a task file.")

    parser.add_argument('--task_path', type=str, help='Path to the task to execute')

    return parser.parse_args()

def main():
    args = parse_arguments()

    task_path = args.task_path

    generation_task = GenerationTask.load_from_json(task_path)


if __name__ == '__main__':
    main()