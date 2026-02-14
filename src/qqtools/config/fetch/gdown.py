import hashlib
import os
import re
from typing import TYPE_CHECKING

from ...qimport import LazyImport

if TYPE_CHECKING:
    import requests
else:
    requests = LazyImport("requests")


__all__ = ["download_from_gdrive_sharelink"]


def calculate_md5(file_path, chunk_size=8192):
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def verify_md5(file_path, expected_md5):
    actual_md5 = calculate_md5(file_path)
    if actual_md5 == expected_md5:

        return True
    else:
        return False


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download_file_from_google_drive(file_id, destination):
    """
    from https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    """
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def download_from_gdrive_sharelink(share_url, dst_path, md5=None):

    if os.path.exists(dst_path) and md5 is not None:
        if verify_md5(dst_path, md5):
            return
        else:
            print(f"{dst_path} exists but MD5 hash not match with {md5}. Re-downloading ...")

    file_id = re.search(r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)", share_url).group(1)
    download_file_from_google_drive(file_id, dst_path)
