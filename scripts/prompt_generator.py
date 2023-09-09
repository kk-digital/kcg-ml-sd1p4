import argparse
import os
import sys
import time
base_directory = os.getcwd()
sys.path.insert(0, base_directory)
from ga.prompt_generator import *


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Prompt Generator CLI tool generates prompts from phrases inside a csv")

    parser.add_argument('--positive-prefix', type=str, help='Prefix phrase to add to positive prompts')
    parser.add_argument('--num-prompts', type=int, default=100, help='Number of prompts to generate')
    parser.add_argument('--csv-phrase-limit', type=int, default=0, help='Number of phrases to use from the csv data')
    parser.add_argument('--csv-path', type=str, help='Full path to the csv path')
    parser.add_argument('--save-embeddings', type=bool, default=True, help='True if prompt embeddings will be saved')
    parser.add_argument('--output', type=str, default="./output/generated_prompts_test",
                        help='Output path for dataset zip containing prompt list npz')
    parser.add_argument('--checkpoint-path', type=str, help='Path to the model checkpoint')
    parser.add_argument('--positive-ratio-threshold', type=int, default=3, help='Threshold ratio of positive/negative to use a phrase for positive prompt')
    parser.add_argument('--negative-ratio-threshold', type=int, default=3, help='Threshold ratio of negative/positive to use a phrase for negative prompt')
    parser.add_argument('--use-threshold', type=bool, default=True, help='True if positive and negative ratio will be used')
    parser.add_argument('--proportional-selection', type=bool, default=False, help='True if proportional selection will be used to get the phrases')

    return parser.parse_args()


def main():
    # Parser the parameters
    args = parse_arguments()
    start_time = time.time()

    # generate and save
    generate_prompts_and_save_to_npz(args.csv_path,
                                     args.csv_phrase_limit,
                                     args.num_prompts,
                                     args.positive_prefix,
                                     args.save_embeddings,
                                     args.checkpoint_path,
                                     args.output,
                                     args.positive_ratio_threshold,
                                     args.negative_ratio_threshold,
                                     args.use_threshold,
                                     args.proportional_selection)

    print("Total Elapsed Time: {0}s".format(time.time() - start_time))


if __name__ == '__main__':
    main()
