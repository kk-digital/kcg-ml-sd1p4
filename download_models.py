import math
import os
import time

import requests
from tqdm import tqdm

from configs.model_config import ModelPathConfig
from utility.utils_logger import logger
from utility.labml.monit import section
from utility.minio.cmd import is_minio_server_accesssible, connect_to_minio_client, download_from_minio

config = ModelPathConfig()


def create_directory_tree_folders(config):
    config.create_paths()


def download_file(url, file_path, description, update_interval=500, chunk_size=4096):
    if not os.path.isfile(file_path):
        def memory2str(mem):
            sizes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
            power = int(math.log(mem, 1024))
            size = sizes[power]
            for _ in range(power):
                mem /= 1024
            if power > 0:
                return f'{mem:.2f}{size}'
            else:
                return f'{mem}{size}'

        with open(file_path, 'wb') as f:
            response = requests.get(url, stream=True)
            total_length = response.headers.get('content-length')
            if total_length is None:
                f.write(response.content)
            else:
                print(f'Downloading {file_path}.', flush=True)
                downloaded, total_length = 0, int(total_length)
                total_size = memory2str(total_length)
                bar_format = '{percentage:3.0f}%|{bar:20}| {desc} [{elapsed}<{remaining}' \
                             '{postfix}]'
                if update_interval * chunk_size * 100 >= total_length:
                    update_interval = 1
                with tqdm(total=total_length, bar_format=bar_format) as bar:
                    counter = 0
                    now_time, now_size = time.time(), downloaded
                    for data in response.iter_content(chunk_size=chunk_size):
                        f.write(data)
                        downloaded += len(data)
                        counter += 1
                        bar.update(len(data))
                        if counter % update_interval == 0:
                            elapsed = time.time() - now_time
                            runtime_downloaded = downloaded - now_size
                            now_time, now_size = time.time(), downloaded

                            cur_size = memory2str(downloaded)
                            speed_size = memory2str(runtime_downloaded / elapsed)
                            bar.set_description(f'{cur_size}/{total_size}')
                            bar.set_postfix_str(f'{speed_size}/s')

                            counter = 0
    else:
        logger.debug(f"{description} already exists.")


if __name__ == "__main__":
    with section("Creating directory tree folders."):
        create_directory_tree_folders(config)

    logger.info("Downloading models. This may take a while.")

    # check if minio server is available
    is_minio_accessible = is_minio_server_accesssible()
    if is_minio_accessible:
        minio_client = connect_to_minio_client()

    with section("Downloading CLIP model"):
        # DOWNLOAD_BASE_CLIP_MODEL
        clip_path = config.get_model('clip/vit-large-patch14', check_existence=False)
        clip_url = r'https://huggingface.co/openai/clip-vit-large-patch14/resolve/refs%2Fpr%2F19/model.safetensors'

        if is_minio_accessible:
            bucket_name = "clip-vit-large-patch14"
            object_name = "model.safetensors"
            download_from_minio(minio_client, bucket_name, object_name, clip_path)
        else:
            download_file(clip_url, clip_path, "CLIP model")

    with section("Downloading Base Diffusion model"):
        # DOWNLOAD_BASE_SD_MODEL

        sd_path = config.get_model('sd/v1-5-pruned-emaonly', check_existence=False)
        sd_url = r'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors'
        if is_minio_accessible:
            bucket_name = "stable-diffusion-models"
            object_name = "v1-5-pruned-emaonly.safetensors"
            download_from_minio(minio_client, bucket_name, object_name, sd_path)
        else:
            download_file(sd_url, sd_path, "Stable Diffusion checkpoint")
