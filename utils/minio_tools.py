from minio.error import S3Error
import os
import json
import urllib3
import json
from minio import Minio
import urllib3
from typing import Tuple, Dict


def collect_files_in_directory(path: str):
    return_files = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            return_files.append(os.path.join(subdir, file))
    return return_files


def upload_local_directory_to_minio(
    local_path, credentials, minio_path, client
):
    try:
        assert os.path.isdir(local_path)
        files = collect_files_in_directory(local_path)
        for file in files:
            remote_path = os.path.join(minio_path, file)
            print(f"Uploading {file} to {remote_path}", flush=True)
            client.fput_object(credentials["bucket"], remote_path, file)
    except S3Error as exc:
        print("error occurred.", exc)


# Default method for loading credentials.
# Expects a file secrets/credentials.json in the root directory
# File must contain: {"endpoint": "xxx", "bucket": "xxx", "username": "xxx", "password": "xxx"}
def load_credentials(
    path: str = r"secrets/credentials.json",
) -> Dict[str, str]:
    assert os.path.exists(path)

    credentials = None
    with open(path, "r", encoding="utf-8") as file:
        credentials = json.load(file)
        file.close()
    return credentials


# Method that opens the session to them Minio storage bucket
def create_session(
    path: str = r"secrets/credentials.json",
) -> Tuple[Minio, Dict[str, str]]:
    credentials = load_credentials(path)

    client = Minio(
        endpoint=credentials["endpoint"],
        access_key=credentials["username"],
        secret_key=credentials["password"],
    )
    # Stripping secret information
    credentials["username"] = ""
    credentials["password"] = ""

    return client, credentials


# Downloads a specified object into a folder as a file
# Skips downloading if a file of the same name already exists
def download_file(client, bucket, object_path, output_path):
    # Get data of an object.
    response = None
    try:
        # Checking if file already exists
        if not os.path.exists(output_path):
            print("Creating empty temporary file...")
            open(output_path, "w").close()
        else:
            print(
                f"File already exists in {output_path}, skipping download..."
            )
            return
        # Requesting object from storage bucket
        print("Getting response...")
        response = client.get_object(bucket, object_path)
        # Read data from response.
        with open(output_path, "wb") as file:
            print("Writing file...")
            file.write(response.data)
            file.close()
            print(f"File written to {output_path}")
    finally:
        if response:
            response.close()
            response.release_conn()


# Streams an object from the bucket
# The object is returned as a HTTPresponse, and must be decoded before use
def stream_file(client, bucket, object_path) -> urllib3.response.HTTPResponse:
    response = None
    try:
        response = client.get_object(bucket, object_path)
        # print(type(response))
        data = response.data
        # print(type(data))
        return data
    finally:
        if response:
            response.close()
            response.release_conn()
