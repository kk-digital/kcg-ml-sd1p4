# Minio Operations

## Minio CLI Tool
### Download an Object
```
usage: minio_operations.py download [-h] [--bucket-name BUCKET_NAME] [--object-name OBJECT_NAME] [--output OUTPUT] [--access-key ACCESS_KEY] [--secret-key SECRET_KEY]

Download an object from a bucket

options:
  -h, --help            show this help message and exit
  --bucket-name BUCKET_NAME
                        Bucket name that contains the object
  --object-name OBJECT_NAME
                        Name of the object to download
  --output OUTPUT       Output path to download the object
  --access-key ACCESS_KEY
                        Your access key to minio server
  --secret-key SECRET_KEY
                        Your secret key to minio server
```

Example:

`python ./scripts/minio_operations.py download --access-key "" --secret-key ""  --bucket-name "test-bucket" --object-name "test-txt.txt" --output "./output"`

### List Buckets
```
usage: minio_operations.py list-buckets [-h] [--access-key ACCESS_KEY] [--secret-key SECRET_KEY]

Gives list of buckets

options:
  -h, --help            show this help message and exit
  --access-key ACCESS_KEY
                        Your access key to minio server
  --secret-key SECRET_KEY
                        Your secret key to minio server
```
Example:

`python ./scripts/minio_operations.py list-buckets --access-key "" --secret-key ""`

### Create Bucket 
```
usage: minio_operations.py create-bucket [-h] [--bucket-name BUCKET_NAME] [--access-key ACCESS_KEY] [--secret-key SECRET_KEY]

Create a bucket

options:
  -h, --help            show this help message and exit
  --bucket-name BUCKET_NAME
                        Bucket name to create
  --access-key ACCESS_KEY
                        Your access key to minio server
  --secret-key SECRET_KEY
                        Your secret key to minio server
```
Example:

`python ./scripts/minio_operations.py create-bucket --access-key "" --secret-key "" --bucket-name "test-bucket"`

### List Objects
```
usage: minio_operations.py list-objects [-h] [--bucket-name BUCKET_NAME] [--access-key ACCESS_KEY] [--secret-key SECRET_KEY]

Get list of objects in a bucket

options:
  -h, --help            show this help message and exit
  --bucket-name BUCKET_NAME
                        Bucket name to get the objects list from
  --access-key ACCESS_KEY
                        Your access key to minio server
  --secret-key SECRET_KEY
                        Your secret key to minio server
```
Example:

`python ./scripts/minio_operations.py list-objects --access-key "" --secret-key "" --bucket-name "test-bucket"`

### Upload an Object
```
usage: minio_operations.py upload [-h] [--bucket-name BUCKET_NAME] [--object-name OBJECT_NAME] [--input INPUT] [--access-key ACCESS_KEY] [--secret-key SECRET_KEY]

Upload an object to a bucket

options:
  -h, --help            show this help message and exit
  --bucket-name BUCKET_NAME
                        Bucket to upload the object
  --object-name OBJECT_NAME
                        Name of the object to upload
  --input INPUT         Path of the object to upload
  --access-key ACCESS_KEY
                        Your access key to minio server
  --secret-key SECRET_KEY
                        Your secret key to minio server
```
Example:

`python ./scripts/minio_operations.py upload --access-key "" --secret-key "" --bucket-name "test-bucket" --object-name "test-txt.txt" --input "./test-txt.txt"`
