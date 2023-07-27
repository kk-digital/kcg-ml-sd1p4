import os
import sys
import configparser
import argparse


config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
parser = argparse.ArgumentParser(description="Setup a config file with the default IO directory structure.")

parser.add_argument("--base_io_directory", type=str, default="./")
parser.add_argument("--base_directory", type=str, default="./")
parser.add_argument("--root_models_prefix", type=str, default="input/models/")
parser.add_argument("--root_outputs_prefix", type=str, default="output/models/")
parser.add_argument("--model_name", type=str, default="v1-5-pruned-emaonly")
parser.add_argument("--clip_model_name", type=str, default="vit-large-patch14")
args = parser.parse_args()

BASE_IO_DIRECTORY = args.base_io_directory
BASE_DIRECTORY = args.base_directory
ROOT_MODELS_PREFIX = args.root_models_prefix
ROOT_OUTPUTS_PREFIX = args.root_outputs_prefix
MODEL_NAME = args.model_name
CLIP_MODEL_NAME = args.clip_model_name

# Please note that using RawConfigParser's set functions, you can assign
# non-string values to keys internally, but will receive an error when
# attempting to write to a file or when you get it in non-raw mode. Setting
# values using the mapping protocol or ConfigParser's set() does not allow
# such assignments to take place.
config.add_section('DEFARGS')
config.set('DEFARGS', 'BASE_IO_DIRECTORY', f'{BASE_IO_DIRECTORY}')
config.set('DEFARGS', 'ROOT_MODELS_PREFIX', f'{ROOT_MODELS_PREFIX}')
config.set('DEFARGS', 'ROOT_OUTPUTS_PREFIX', f'{ROOT_OUTPUTS_PREFIX}')
config.set('DEFARGS', 'MODEL_NAME', f"{MODEL_NAME}")
config.set('DEFARGS', 'CLIP_MODEL_NAME', f'{CLIP_MODEL_NAME}')
config.set('DEFARGS', 'CHECKPOINT', "${MODEL_NAME}.safetensors")

# Writing our configuration file to 'example.cfg'
with open('config.cfg', 'w') as configfile:
    config.write(configfile)