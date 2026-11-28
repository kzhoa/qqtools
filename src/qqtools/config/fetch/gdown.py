import hashlib
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from ...qimport import LazyImport
from .._atomic_file import atomic_output_file

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
    return calculate_md5(file_path) == expected_md5.lower()


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response: "requests.Response", destination: str | Path, md5: str | None = None) -> None:
    response.raise_for_status()
    digest = hashlib.md5() if md5 is not None else None
    with atomic_output_file(Path(destination)) as staged:
        with staged.open("wb") as stream:
            for chunk in response.iter_content(32768):
                stream.write(chunk)
                if digest is not None:
                    digest.update(chunk)
        if digest is not None and digest.hexdigest() != md5.lower():
            raise ValueError(f"MD5 mismatch for {destination}")


def download_file_from_google_drive(file_id: str, destination: str | Path, md5: str | None = None) -> None:
    url = "https://docs.google.com/uc"
    params = {"export": "download", "id": file_id, "confirm": "1"}
    with requests.Session() as session:
        with session.get(url, params=params, stream=True) as response:
            token = get_confirm_token(response)
            if not token:
                save_response_content(response, destination, md5)
                return
            response.raise_for_status()
        with session.get(url, params={**params, "confirm": token}, stream=True) as confirmed:
            save_response_content(confirmed, destination, md5)


def download_from_gdrive_sharelink(share_url: str, dst_path: str | Path, md5: str | None = None) -> None:
    """Publish the downloaded file after a successful response and optional MD5 check."""
    if os.path.exists(dst_path) and md5 is not None:
        if verify_md5(dst_path, md5):
            return
        print(f"{dst_path} exists but MD5 hash not match with {md5}. Re-downloading ...")

    match = re.search(r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)", share_url)
    if match is None:
        raise ValueError("Expected a Google Drive file share URL")
    download_file_from_google_drive(match.group(1), dst_path, md5)
