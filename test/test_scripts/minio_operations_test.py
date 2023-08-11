# Commented for now since this test needs ACCESS KEY and SECRET KEY
# import sys
# base_directory = "./"
# sys.path.insert(0, base_directory)
# from utility.minio import cmd
#
# cmd.MINIO_ADDRESS = "localhost:9000"
#
# ACCESS_KEY = ""
# SECRET_KEY = ""
#
# def test_list_buckets():
#     expected_bucket_list_len = 1
#     if cmd.is_minio_server_accesssible() and ACCESS_KEY is not None and SECRET_KEY is not None:
#         client = cmd.connect_to_minio_client(ACCESS_KEY, SECRET_KEY)
#         buckets = cmd.get_list_of_buckets(client)
#
#         assert len(buckets) > expected_bucket_list_len
#     else:
#         return
#
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

