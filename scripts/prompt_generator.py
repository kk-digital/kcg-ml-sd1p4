import argparse
import os
import sys

base_directory = os.getcwd()
sys.path.insert(0, base_directory)
from ga.prompt_generator import *

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Prompt Generator CLI tool")

    parser.add_argument('--num-prompts', type=int, default=100, help='Number of prompts to generate')
    parser.add_argument('--num-phrases', type=int, default=12, help='Number of phrases per prompt')
    parser.add_argument('--output', type=str, default="./output", help='Output path for prompt list json')

    return parser.parse_args()


def main():
    # Parser the parameters
    args = parse_arguments()
    start_time = time.time()

    # generate and save
    generate_prompts_and_save_to_json(args.num_prompts, args.num_phrases, args.output)

    print("Total Elapsed Time: {0}s".format(time.time() - start_time))

if __name__ == '__main__':
    main()
