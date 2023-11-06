from minio import Minio
import os
import requests
from .progress import Progress
from utility.utils_logger import logger

# TODO: remove hardcode in the future
#  use config file
MINIO_ADDRESS = "192.168.3.5:9000"


def connect_to_minio_client(access_key=None, secret_key=None):
    print("Connecting to minio client...")
    client = Minio(MINIO_ADDRESS, access_key, secret_key, secure=False)
    print("Successfully connected to minio client...")
    return client


def is_minio_server_accesssible():
    print("Checking if minio server is accessible...")
    try:
        r = requests.head("http://" + MINIO_ADDRESS + "/minio/health/live", timeout=5)
    except:
        print("Minio server is not accessible...")
        return False

    return r.status_code == 200


def download_from_minio(client, bucket_name, object_name, output_path):
    if not os.path.isfile(output_path):
        client.fget_object(bucket_name, object_name, output_path, progress=Progress())
    else:
        logger.info(f"{object_name} already exists.")


def get_list_of_buckets(client):
    buckets = client.list_buckets()
    for bucket in buckets:
        print("Bucket: {0}: {1}".format(bucket.name, bucket.creation_date))

    return buckets


def check_if_bucket_exists(client, bucket_name):
    if client.bucket_exists(bucket_name):
        print("{0} exists".format(bucket_name))
        return True

    print("{0} does not exists".format(bucket_name))
    return False


def create_bucket(client, bucket_name):
    client.make_bucket(bucket_name)
    print("Bucket: {0} successfully created...".format(bucket_name))


def remove_bucket(client, bucket_name):
    client.remove_bucket(bucket_name)
    print("Bucket: {0} successfully deleted...".format(bucket_name))


def get_list_of_objects(client, bucket_name):
    objects = client.list_objects(bucket_name)
    print("Objects inside bucket {0}".format(bucket_name))
    for obj in objects:
        print(obj.object_name)

    return objects


def upload_from_file(client, bucket_name, object_name, file_path):
    result = client.fput_object(bucket_name, object_name, file_path)
    print(
        "created {0} object; etag: {1}, version-id: {2}".format(
            result.object_name, result.etag, result.version_id,
        ),
    )


def upload_data(client, bucket_name, object_name, data):
    result = client.put_object(
        bucket_name, object_name, data, length=-1, part_size=10 * 1024 * 1024,
    )
    print(
        "created {0} object; etag: {1}, version-id: {2}".format(
            result.object_name, result.etag, result.version_id,
        ),
    )


def remove_an_object(client, bucket_name, object_name):
    # Remove object.
    client.remove_object(bucket_name, object_name)


def is_object_exists(client, bucket_name, object_name):
    result = client.stat_object(bucket_name, object_name)
    print(
        "last-modified: {0}, size: {1}".format(
            result.last_modified, result.size,
        ),
    )

    if result.object_name != "":
        return True

    return False
