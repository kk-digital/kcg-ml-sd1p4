import argparse
import os
import sys

base_directory = os.getcwd()
sys.path.insert(0, base_directory)
from utility.minio import cmd


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Minio operations CLI tool")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Add sub-commands
    # download subcommand
    download_parser = subparsers.add_parser("download", help="Use to download an object",
                                         description="Download an object from a bucket")
    download_parser.add_argument('--bucket-name', type=str, default=None, help='Bucket name that contains the object')
    download_parser.add_argument('--object-name', type=str, default=None, help='Name of the object to download')
    download_parser.add_argument('--output', type=str, default=None, help='Output path to download the object')
    download_parser.add_argument('--access-key', type=str, default=None, help='Your access key to minio server')
    download_parser.add_argument('--secret-key', type=str, default=None, help='Your secret key to minio server')

    #  list-buckets subcommand
    list_buckets_parser = subparsers.add_parser("list-buckets", help="Get list of buckets. Needs ACCESS_KEY and SECRET_KEY.",
                                            description="Gives list of buckets")
    list_buckets_parser.add_argument('--access-key', type=str, default=None, help='Your access key to minio server')
    list_buckets_parser.add_argument('--secret-key', type=str, default=None, help='Your secret key to minio server')

    # create-bucket subcommand
    create_bucket_parser = subparsers.add_parser("create-bucket", help="Use to create a bucket. Needs ACCESS_KEY and SECRET_KEY.",
                                         description="Create a bucket")
    create_bucket_parser.add_argument('--bucket-name', type=str, default=None, help='Bucket name to create')
    create_bucket_parser.add_argument('--access-key', type=str, default=None, help='Your access key to minio server')
    create_bucket_parser.add_argument('--secret-key', type=str, default=None, help='Your secret key to minio server')

    # list_objects subcommand
    list_objects_parser = subparsers.add_parser("list-objects", help="Get list of objects in a bucket. Needs ACCESS_KEY and SECRET_KEY.",
                                            description="Get list of objects in a bucket")
    list_objects_parser.add_argument('--bucket-name', type=str, default=None, help='Bucket name to get the objects list from')
    list_objects_parser.add_argument('--access-key', type=str, default=None, help='Your access key to minio server')
    list_objects_parser.add_argument('--secret-key', type=str, default=None, help='Your secret key to minio server')

    # upload an object subcommand
    upload_parser = subparsers.add_parser("upload", help="Use to upload an object. Needs ACCESS_KEY and SECRET_KEY.",
                                            description="Upload an object to a bucket")
    upload_parser.add_argument('--bucket-name', type=str, default=None, help='Bucket to upload the object')
    upload_parser.add_argument('--object-name', type=str, default=None, help='Name of the object to upload')
    upload_parser.add_argument('--input', type=str, default=None, help='Path of the object to upload')
    upload_parser.add_argument('--access-key', type=str, default=None, help='Your access key to minio server')
    upload_parser.add_argument('--secret-key', type=str, default=None, help='Your secret key to minio server')

    return parser.parse_args()


def main():
    # Parser the parameters
    args = parse_arguments()

    # check if server is acessible
    is_accessible = cmd.is_minio_server_accesssible()

    if not is_accessible:
        raise Exception("Minio server not accessible")

    # connect to minio
    client = cmd.connect_to_minio_client(args.access_key, args.secret_key)

    if args.command == "download":
        cmd.download_from_minio(client, args.bucket_name, args.object_name, os.path.join(args.output, args.object_name))
    elif args.command == "list-buckets":
        cmd.get_list_of_buckets(client)
    elif args.command == "create-bucket":
        cmd.create_bucket(client, args.bucket_name)
    elif args.command == "list-objects":
        cmd.get_list_of_objects(client, args.bucket_name)
    elif args.command == "upload":
        cmd.upload_from_file(client, args.bucket_name, args.object_name, args.input)


if __name__ == '__main__':
    main()
