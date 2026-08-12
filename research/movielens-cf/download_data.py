"""Download and validate MovieLens 1M into the gitignored local data directory."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
EXPECTED_MD5 = "c4d9eecfca2ab87c1945afe126590906"
DESTINATION = Path(__file__).with_name("data")


def main() -> None:
    DESTINATION.mkdir(exist_ok=True)
    archive = DESTINATION / "ml-1m.zip"
    if not archive.exists():
        with urlopen(URL) as response, archive.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    checksum = hashlib.md5(archive.read_bytes(), usedforsecurity=False).hexdigest()
    if checksum != EXPECTED_MD5:
        raise ValueError(f"MovieLens archive checksum mismatch: {checksum}")
    data_dir = DESTINATION / "ml-1m"
    required = [data_dir / name for name in ("ratings.dat", "movies.dat", "users.dat", "README")]
    if not all(path.exists() for path in required):
        with ZipFile(archive) as bundle:
            bundle.extractall(DESTINATION)
    print(data_dir)


if __name__ == "__main__":
    main()
