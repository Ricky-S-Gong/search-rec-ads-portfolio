"""Download and verify the official KuaiRand-1K archive without committing it."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen


URL = "https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz"
MD5 = "6b0b9c8222d67fcd4c676218edca3f1f"


def checksum(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download KuaiRand-1K")
    parser.add_argument("--destination", type=Path, default=Path(__file__).with_name("data") / "raw" / "KuaiRand-1K.tar.gz")
    args = parser.parse_args()
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    if not args.destination.exists():
        partial = args.destination.with_suffix(args.destination.suffix + ".part")
        with urlopen(URL) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        partial.replace(args.destination)
    actual = checksum(args.destination)
    if actual != MD5:
        raise ValueError(f"MD5 mismatch: expected {MD5}, got {actual}")
    print(args.destination)


if __name__ == "__main__":
    main()
