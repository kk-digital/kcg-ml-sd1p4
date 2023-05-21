import os
import datetime
import time
import argparse

import libtorrent as lt
import requests as rq

from typing import Literal, Union
from model_list_type import model_list, Model

def check_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

    return path

DESTINATION = '/tmp/input/models/'
OUTPUT_DIR = '/output/tmp'
LOGS_DIR = '/tmp/logs'


def download_via_magnet_link(magnet_link: str, destination: str):
    ses = lt.session()
    h = lt.add_magnet_uri(ses, magnet_link, {'save_path': destination})

    print("Downloading magnet link...")

    while not h.is_seed():
        s = h.status()
        print(f"Progress: {s.progress * 100:.2f}%")
        time.sleep(1)

    print(f"Download complete. File saved to: {destination}/{h.name()}")

def download_via_direct_link(direct_link: str, destination: str, filename: str = None):
    print("Downloading direct link...")

    r = rq.get(direct_link, allow_redirects=True, stream=True) ## print progress below

    total_length = int(r.headers.get('content-length')) if 'content-length' in r.headers else None

    with open(os.path.join(destination, filename), 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            f.write(chunk)

            if chunk and total_length:
                print(f"Progress: {round(len(chunk) / total_length * 100, 2)}%")
            if chunk and not total_length:
                print(f"Progress: {round(len(chunk) / 1024 / 1024, 2)}MB")

    print(f"Download complete. File saved to: {destination}/{filename}")

def download_model(
        model: Model,
        via: Union[Literal['direct'], Literal['magnet']] = 'direct_link',
        destination: str = '/tmp/input/models/',
    ):

    if via == 'direct':
        download_via_direct_link(model['direct_link'], destination)
    elif via == 'magnet':
        download_via_magnet_link(model['magnet_link'], destination, model['filename'])
    

def show_models():
    for model in model_list:
        direct_link = f"Direct link: {model['direct_link']}\n" if 'direct_link' in model else ''
        magnet_link = f"Magnet link: {model['magnet_link']}\n" if 'magnet_link' in model else ''
        print(f"{model['filename']}\n{model['description']}\n{direct_link}{magnet_link}")

def main():
    parser = argparse.ArgumentParser(description='Download model')

    parser.add_argument('--list-models', type=str, help='List models', action='store_true')

    parser.add_argument('--model', type=str, help='Filename of model to download')

    parser.add_argument('--destination', type=str, help='Destination to save model to', default=DESTINATION)
    
    args = parser.parse_args()

    if args.list_models:
        show_models()
        return
    
    if args.model:
        model_link = next((model for model in model_list if model['filename'] == args.model), None)

        if model_link:
            check_dir(args.destination or DESTINATION)

            if os.path.exists(os.path.join(args.destination or DESTINATION, model_link)):
                print('Model already exists')
                return

            download_model(model_link, via='direct', destination=args.destination or DESTINATION)
        else:
            print('Model not found')

        return
    

if __name__ == '__main__':
    main()