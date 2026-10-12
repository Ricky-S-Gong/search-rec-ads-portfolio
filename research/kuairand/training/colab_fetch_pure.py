"""Manually run in the Colab host kernel to cache the verified shared Drive files.

Requires Colab's preinstalled google-auth/google-api-python-client. This is a
host-only download utility, not part of the locked training environment.
"""

import hashlib
import io
import json
from pathlib import Path


def fetch_pure(destination, specification):
    """Authenticate the notebook owner and download exact, hash-pinned file IDs."""
    import google.auth
    from google.colab import auth
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    spec = json.loads(Path(specification).read_text())
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    auth.authenticate_user()
    credentials, _ = google.auth.default()
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    for name, entry in spec["files"].items():
        target = destination / name
        if target.exists():
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest != entry["sha256"]:
                raise ValueError(
                    f"Existing cache differs: {target}; choose a fresh cache directory"
                )
            print(f"Verified cached {name}")
            continue
        metadata = (
            service.files()
            .get(fileId=entry["id"], fields="id,name,size", supportsAllDrives=True)
            .execute()
        )
        if metadata["name"] != name or int(metadata["size"]) != entry["bytes"]:
            raise ValueError(f"Published source changed: {name}; re-audit before training")
        temporary = target.with_suffix(target.suffix + ".partial")
        with io.FileIO(temporary, "wb") as stream:
            downloader = MediaIoBaseDownload(
                stream, service.files().get_media(fileId=entry["id"], supportsAllDrives=True)
            )
            done = False
            while not done:
                _, done = downloader.next_chunk()
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != entry["sha256"]:
            temporary.unlink()
            raise ValueError(f"Published source changed: {name}; re-audit before training")
        temporary.replace(target)
        print(f"Downloaded and verified {name}")
    return destination
