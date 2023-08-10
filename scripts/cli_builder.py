import argparse
import sys
from typing import Union, Callable

sys.path.append("./")

from stable_diffusion import CHECKPOINT_PATH
from stable_diffusion_reference.cli import check_folder_existence, check_file_existence


class CLI:
    def __init__(self, description: str):
        self.parser = argparse.ArgumentParser(description=description)
        self.checkers = []

    def _required(self, args, name, message):
        if not getattr(args, name):
            print(message)
            exit(1)

    def list_models(self):
        self.parser.add_argument(
            '--list_models',
            help='List models',
            action='store_true'
        )

        return self

    def model(self):
        self.parser.add_argument(
            '--model',
            help='Filename of model to download, showed in --list-models',
            default='v1-5-pruned-emaonly.safetensors'
        )

        return self

    def output(self, default='./output', check_exists: Union[bool, Callable] = True):
        self.parser.add_argument('--output',
                                 help='Path to the output directory (default: %(default)s)',
                                 default=default
                                 )

        if check_exists:
            check_fn = check_exists if callable(check_exists) else lambda _: check_exists

            self.checkers.append(
                lambda args: check_fn(args) and check_folder_existence(args.output)
            )

        return self

    def prompt_prefix(self):
        self.parser.add_argument(
            '--prompt_prefix',
            type=str,
            default="A woman with flowers in her hair in a courtyard, in the style of",
            help='Prefix for the prompt, must end with "in the style of" (default: %(default)s)'
        )

        return self

    def artist_file(self, check_exists: Union[bool, Callable] = True):
        self.parser.add_argument(
            '--artist_file',
            type=str,
            default='./input/artists.txt',
            help='Path to the file containing the artists, each on a line (default: \'%(default)s\')'
        )

        if check_exists:
            check_fn = check_exists if callable(check_exists) else lambda _: check_exists

            self.checkers.append(
                lambda args: check_fn(args) and check_file_existence(args.artist_file)
            )

        return self

    def checkpoint_path(self, check_exists: Union[bool, Callable] = True):
        self.parser.add_argument(
            '--checkpoint_path',
            type=str,
            default=CHECKPOINT_PATH,
            help='Path to the checkpoint file (default: \'./input/model/v1-5-pruned-emaonly.safetensors\')'
        )

        if check_exists:
            check_fn = check_exists if callable(check_exists) else lambda _: check_exists

            self.checkers.append(
                lambda args: check_fn(args) and check_file_existence(args.checkpoint_path)
            )

        return self

    def sampler(self):
        self.parser.add_argument(
            '--sampler',
            type=str,
            default='ddim',
            help='Name of the sampler to use (default: %(default)s)'
        )

        return self

    def steps(self):
        self.parser.add_argument(
            '--steps',
            type=int,
            default=50,
            help='Number of steps to use (default: %(default)s)'
        )

        return self

    def batch_size(self):
        self.parser.add_argument(
            '--batch_size',
            type=int,
            default=1,
            help='How many images to generate at once (default: %(default)s)'
        )

        return self

    def image_width(self):
        self.parser.add_argument(
            '--image_width',
            type=int,
            default=512,
            help='Generate image width (default: %(default)s)'
        )

        return self

    def image_height(self):
        self.parser.add_argument(
            '--image_height',
            type=int,
            default=512,
            help='Generate image height (default: %(default)s)'
        )

        return self

    def num_datasets(self):
        self.parser.add_argument(
            '--num_datasets',
            type=int,
            default=1,
            help='Number of datasets to generate (default: %(default)s)'
        )

        return self
    def num_images(self):
        self.parser.add_argument(
            '--num_images',
            type=int,
            default=1,
            help='How many images to generate (default: %(default)s)'
        )

        return self

    def seed(self):
        self.parser.add_argument(
            '--seed',
            type=str,
            default='',
            help='Seed for the image generation (default: %(default)s)'
        )

        return self

    def num_iterations(self):
        self.parser.add_argument(
            '--num_iterations',
            type=int,
            default=10,
            help='How many times to iterate the generation of a batch of images (default: %(default)s)'
        )

        return self

    def noise_seed(self):
        self.parser.add_argument(
            '--noise_seed',
            type=int,
            default=42,
            help='The noise seed to be fed to the sampling process (default: %(default)s)'
        )

        return self

    def prompts_file(self, check_exists: Union[bool, Callable] = True, required: bool = True):
        self.parser.add_argument(
            '--prompts_file',
            type=str,
            default='./input/prompts.txt',
            help='Path to the file containing the prompts, each on a line (default: \'%(default)s\')'
        )

        if check_exists:
            check_fn = lambda args: (check_exists(args) if callable(check_exists) else check_exists) and required

            self.checkers.append(
                lambda args: check_fn(args) and check_file_existence(args.prompts_file)
            )

        return self

    def strength(self):
        self.parser.add_argument(
            '--strength',
            type=float,
            default=0.75,
            help='Strength for noise: 1.0 corresponds to full destruction of information in init image (default: %(default)s)'
        )

        return self

    def prompt(self):
        self.parser.add_argument(
            '--prompt',
            type=str,
            nargs="?",
            help='The prompt to render',
            default="a painting of a cute monkey playing guitar"
        )

        return self

    def orig_img(self, check_exists: Union[bool, Callable] = True, required: bool = True):
        self.parser.add_argument(
            "--orig_img",
            type=str,
            nargs="?",
            help="path to the input image"
        )

        if required:
            self.checkers.append(
                lambda args: self._required(args, 'orig_img',
                                            '\nERROR: Please specify the path to the input image with --orig_img')
            )

        if check_exists:
            check_fn = check_exists if callable(check_exists) else lambda _: check_exists

            self.checkers.append(
                lambda args: check_fn(args) and check_file_existence(args.orig_img)
            )

        return self

    def cfg_scale(self):
        self.parser.add_argument(
            "--cfg_scale",
            type=float,
            default=5.0,
            help="unconditional guidance scale: "
                 "eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))"
        )

        return self

    def force_cpu(self):
        self.parser.add_argument(
            "--force_cpu",
            action='store_true',
            help="force CPU usage"
        )

        return self

    def cuda_device(self):
        self.parser.add_argument(
            "--cuda_device",
            type=str,
            default="cuda:0",
            help="cuda device to use for generation"
        )

        return self

    def flash(self):
        self.parser.add_argument(
            "--flash",
            action='store_true',
            help="whether to use flash attention"
        )

        return self

    def low_vram(self):
        self.parser.add_argument(
            "--low_vram",
            action='store_true',
            help="limit VRAM usage"
        )

        return self

    def output_metadata(self):
        self.parser.add_argument(
            "--output_metadata",
            action='store_true',
            help="outputs the metadata"
        )

        return self

    def parse(self):
        args = self.parser.parse_args()

        for checker in self.checkers:
            checker(args)

        return args

    def num_phrases(self):
        self.parser.add_argument(
            '--num-phrases',
            type=int,
            nargs="?",
            help='The number of phrases for the prompt to generate',
            default=12
        )

        return self
