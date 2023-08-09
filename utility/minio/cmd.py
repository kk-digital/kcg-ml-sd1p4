from minio import Minio
import os
import requests
from .progress import Progress
from utility.utils_logger import logger

# TODO: remove hardcode in the future
#  use config file
MINIO_ADDRESS = "192.168.3.5:9000"


def connect_to_minio_client():
    print("Connecting to minio client...")
    client = Minio(MINIO_ADDRESS, secure=False)
    print("Successfully connected to minio client...")
    return client


def download_from_minio(client, bucket_name, object_name, output_path):
    if not os.path.isfile(output_path):
        client.fget_object(bucket_name, object_name, output_path, progress=Progress())
    else:
        logger.info(f"{object_name} already exists.")


def is_minio_server_accesssible():
    print("Checking if minio server is accessible...")
    try:
        r = requests.head("http://" + MINIO_ADDRESS + "/minio/health/live", timeout=5)
    except:
        print("Minio server is not accessible...")
        return False

    return r.status_code == 200