import os
from collections import defaultdict

from utility.utils_logger import logger


class DefaultPaths:
    ROOT_DIRECTORY = './'
    INPUT_DIRECTORY = 'input/'
    OUTPUT_DIRECTORY = 'output/'
    MODELS_DIRECTORY = 'model/'

    MODELS = {
        'clip': {'vit-large-patch14',
                 'txt_emb', 'txt_emb_tokenizer', 'txt_emb_model',
                 'img_enc', 'img_enc_processor', 'img_enc_vision'},
        'sd': {'v1-5-pruned-emaonly', 'latent_diffusion'},
        'unet': {'unet'},
        'vae': {'vae', 'decoder', 'encoder'},
        'aesthetic_scorer': {'aesthetic_scorer'}
    }


class ModelPathConfig:
    DEFAULT_EXTENSION = '.safetensors'

    def __init__(self, root_directory=DefaultPaths.ROOT_DIRECTORY, check_existence=True):
        self.check_path_existence = check_existence

        self.root_directory = root_directory
        self.input_directory = os.path.join(self.root_directory, DefaultPaths.INPUT_DIRECTORY)
        self.models_directory = os.path.join(self.input_directory, 'model/')

        self.output_directory = os.path.join(self.root_directory, DefaultPaths.OUTPUT_DIRECTORY)

        self._validate_paths([self.root_directory, self.input_directory, self.output_directory, self.models_directory])

    @staticmethod
    def _validate_paths(paths):
        for path in paths:
            if not os.path.exists(path):
                logger.warning(f"Path {path} does not exist.")
                raise FileNotFoundError(f"Path {path} does not exist.")

    @staticmethod
    def _warning_or_error(text, raise_error=False):
        logger.warning(text)
        if raise_error:
            raise FileNotFoundError(text)

    def _get_paths(self, *paths, extension=DEFAULT_EXTENSION, raise_error=False, check_existence=None,
                   include_extension=True):
        """
        :param Union[str, List[str]] paths: path(s) to the model
        :param bool to_dict: whether to return a dict or not
        :param Union[str, List[str]] extension: extension(s) of the model
        :param bool raise_error: whether to raise an error if the path doesn't exist
        :param bool check_existence: whether to check if the path exists

        If only one extension is given, it will be used for all paths. If multiple extensions are given, the number of
        extensions must match the number of paths.


        :return: the path(s) to the model(s)
        """

        check_existence = self.check_path_existence if check_existence is None else check_existence

        if isinstance(extension, str):
            extension = [extension] * len(paths)

        if len(paths) != len(extension):
            logger.warning("Number of paths and extensions must be the same.")
            return None

        results = []
        for path, ext in zip(paths, extension):
            model_dir, model_name = os.path.split(path)

            if model_dir not in DefaultPaths.MODELS.keys():
                logger.debug(f"Model directory {model_dir} is not valid.", raise_error)
                continue

            if model_name not in DefaultPaths.MODELS.get(model_dir, {}):
                logger.debug(f"Model {model_name} wasn't set up in the config for {model_dir}.")
                continue

            full_path = os.path.join(self.models_directory, model_dir, model_name)
            if include_extension:
                full_path = os.path.join(full_path, model_name + ext)

            if check_existence and not os.path.exists(full_path):
                self._warning_or_error(f"Path {full_path} does not exist.")
            else:
                results.append((model_name, full_path))

        return results

    @staticmethod
    def _format_results(results, to_dict):
        if not results:
            return None

        if to_dict:
            return {key: value for key, value in results}

        return results[0][1] if len(results) == 1 else [value for _, value in results]

    def get_model(self, *paths, to_dict=False, extension=DEFAULT_EXTENSION, raise_error=False, check_existence=None):
        """
        :return: the path to the model with the model name and extension
        """
        results = self._get_paths(*paths, extension=extension, raise_error=raise_error, check_existence=check_existence)
        return self._format_results(results, to_dict)

    def get_model_folder_path(self, *paths, to_dict=False, raise_error=False, check_existence=None):
        """
        :return: the path to the model without the model name and extension
        """
        results = self._get_paths(*paths, raise_error=raise_error, check_existence=check_existence,
                                  include_extension=False)
        return self._format_results(results, to_dict)

    def get_model_path(self):
        return self.models_directory

    def get_input_path(self):
        return self.input_directory

    def get_output_path(self):
        return self.output_directory

    def list_models(self):
        tree = defaultdict(list)
        for model_type, models in DefaultPaths.MODELS.items():
            for model_name in models:
                for ext in ['.safetensors', '.pt', '.ckpt', '.bin']:
                    full_path = self.get_model(f'{model_type}/{model_name}', extension=ext)
                    if full_path is not None:
                        break
                tree[model_type].append((model_name, full_path))

        for model_type, nodes in tree.items():
            print(f"{model_type}:")
            for model_name, path in nodes:
                print(f" ├─ {model_name}: {path}")

    def check_paths_exist(self):
        all_paths_exist = True
        for model_type, models in DefaultPaths.MODELS.items():
            for model_name in models:
                full_path = self.get_model(f'{model_type}/{model_name}')
                if full_path is None:
                    all_paths_exist = False
                    logger.warning(f"Path {full_path} does not exist.")
        return all_paths_exist

    def create_paths(self):
        # Create input, output and models directories
        os.makedirs(self.input_directory, exist_ok=True)
        os.makedirs(self.output_directory, exist_ok=True)
        os.makedirs(self.models_directory, exist_ok=True)

        # Create models tree folders
        for model_type, models in DefaultPaths.MODELS.items():
            for model_name in models:
                full_path = os.path.join(self.models_directory, model_type, model_name)
                os.makedirs(full_path, exist_ok=True)
        logger.info("All model paths have been created.")

    def __repr__(self):
        return f"""IODirectoryTree:
                - {self.root_directory}
                -- model's root dir: {self.models_directory}
                --- stable diffusion model checkpoint: {self.get_model('sd/v1-5-pruned-emaonly')}
                --- stable diffusion submodels dir: {self.models_directory + 'sd'}
                ---- unet model: {self.get_model('sd/unet')}
                ---- autoencoder model: {self.get_model('sd/autoencoder')}
                ---- encoder model: {self.get_model('sd/encoder')}
                ---- decoder model: {self.get_model('sd/decoder')}
                ---- latent_diffusion model: {self.get_model('sd/latent_diffusion')}
                --- clip models' root dir: {self.models_directory + 'clip'}
                ---- clip text embedder submodels dir: {self.models_directory + 'clip/text_embedder'}
                ----- clip text embedder checkpoint path: {self.get_model('clip/text_embedder')}
                ----- tokenizer: {self.get_model('clip/tokenizer')}
                ----- text model: {self.get_model('clip/text_model')}
                ---- clip image encoder submodels dir: {self.models_directory + 'clip/image_encoder'}
                ----- clip image encoder checkpoint path: {self.get_model('clip/image_encoder')}
                ----- vision_model: {self.get_model('clip/vision_model')}
                ----- image_processor: {self.get_model('clip/image_processor')}
                --- model's outputs dir: {self.get_output_path()}
            """
