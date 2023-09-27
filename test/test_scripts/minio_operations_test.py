# # Commented for now since this test needs ACCESS KEY and SECRET KEY
# import sys
# base_directory = "./"
# sys.path.insert(0, base_directory)
# from utility.minio import cmd, progress
# import boto3
# from tqdm import tqdm
# import io
# from botocore.client import Config
# from PIL import Image
# import requests
# ACCESS_KEY =
# SECRET_KEY =
# cmd.MINIO_ADDRESS =
#
# def test_list_buckets():
#     expected_bucket_list_len = 1
#     if cmd.is_minio_server_accesssible() and ACCESS_KEY is not None and SECRET_KEY is not None:
#         client = cmd.connect_to_minio_client(ACCESS_KEY, SECRET_KEY)
#         buckets = cmd.get_list_of_buckets(client)
#         print(buckets)
#         assert len(buckets) > expected_bucket_list_len
#     else:
#         return
#
#
# def test_upload_data():
#     bucket_name = "datasets"
#     file_name = "icons/test_image.jpg"
#
#     url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#     image = Image.open(requests.get(url, stream=True).raw)
#     img_byte_arr = io.BytesIO()
#     image.save(img_byte_arr, format='JPEG')
#     img_byte_arr.seek(0)
#
#
#     if cmd.is_minio_server_accesssible() and ACCESS_KEY is not None and SECRET_KEY is not None:
#         client = cmd.connect_to_minio_client(ACCESS_KEY, SECRET_KEY)
#         cmd.upload_data(client, bucket_name, file_name, img_byte_arr)
#     else:
#         return
#
# # def test_list_zip_object_buckets():
# #     expected_bucket_list_len = 1
# #     if cmd.is_minio_server_accesssible() and ACCESS_KEY is not None and SECRET_KEY is not None:
# #         client = cmd.connect_to_minio_client(ACCESS_KEY, SECRET_KEY)
# #
# #         objects = client.list_objects(bucket_name)
# #         print("Objects inside bucket {0}".format(bucket_name))
# #         for obj in objects:
# #             print(obj.object_name)
# #
# #     else:
# #         return
#
# def test_download_object_from_zip():
#     expected_bucket_list_len = 1
#     if cmd.is_minio_server_accesssible() and ACCESS_KEY is not None and SECRET_KEY is not None:
#         client = cmd.connect_to_minio_client(ACCESS_KEY, SECRET_KEY)
#
#         client.fget_object(bucket_name="pixel-art-dataset",
#                            object_name="pixel-art-000001.zip/pixel-art-000001/manifest.json",
#                            file_path="./manifest.json",
#                            request_headers={"x-minio-extract": "true"},
#                            progress=progress.Progress())
#
#     else:
#         return
#
# def test_bucket_operations():
#     test_bucket = "test-bucket-create"
#     if cmd.is_minio_server_accesssible() and ACCESS_KEY is not None and SECRET_KEY is not None:
#         client = cmd.connect_to_minio_client(ACCESS_KEY, SECRET_KEY)
#
#         # create
#         cmd.create_bucket(client, test_bucket)
#
#         # check fi exists
#         is_exist = cmd.check_if_bucket_exists(client, test_bucket)
#         assert is_exist == True
#
#         # remove
#         cmd.remove_bucket(client, test_bucket)
#
#     else:
#         return
#
#
# def test_boto3_minio():
#     # create s3 client, connect to minio server
#     s3 = boto3.client('s3',
#                       endpoint_url="http://localhost:9000",
#                       aws_access_key_id=ACCESS_KEY,
#                       aws_secret_access_key=SECRET_KEY,
#                       )
#
#     # add x-minio-extract header, needed to get info inside zip file
#     def _add_header(request, **kwargs):
#         request.headers.add_header('x-minio-extract', 'true')
#
#     event_system = s3.meta.events
#     event_system.register_first('before-sign.s3.*', _add_header)
#
#     # List zip contents
#     response = s3.list_objects_v2(Bucket="pixel-art-dataset", Prefix="pixel-art-000001.zip/")
#     for item in response["Contents"]:
#         print(item["Key"])
#
#     # download a file inside the zip
#     s3_download(s3, 'pixel-art-dataset', 'pixel-art-000001.zip/pixel-art-000001/manifest.json', './manifest.json')
#
# def s3_download(s3_client, s3_bucket, s3_object_key, file_name):
#     meta_data = s3_client.head_object(Bucket=s3_bucket, Key=s3_object_key)
#     total_length = int(meta_data.get('ContentLength', 0))
#     with tqdm(total=total_length,  desc=f'source: s3://{s3_bucket}/{s3_object_key}', bar_format="{percentage:.1f}%|{bar:25} | {rate_fmt} | {desc}",  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
#         s3_client.download_file(s3_bucket, s3_object_key, file_name, Callback=pbar.update)