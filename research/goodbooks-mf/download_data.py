"""Download the official Goodreads Poetry files into the gitignored data directory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre"
FILES = (
    "goodreads_interactions_poetry.json.gz",
    "goodreads_books_poetry.json.gz",
)
DESTINATION = Path(__file__).with_name("data") / "raw"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for name in FILES:
        destination = DESTINATION / name
        if not destination.exists():
            print(f"Downloading {name} ...")
            with urlopen(f"{BASE_URL}/{name}") as response, destination.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        checksums[name] = sha256(destination)
    (DESTINATION / "source_manifest.json").write_text(
        json.dumps(
            {
                "source": "UCSD Goodreads Book Graph",
                "subset": "Poetry",
                "files": checksums,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(DESTINATION)


if __name__ == "__main__":
    main()
