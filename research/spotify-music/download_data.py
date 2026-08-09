"""Download the public Kaggle dataset into the gitignored research data folder."""

from pathlib import Path
import subprocess

destination = Path(__file__).with_name("data")
destination.mkdir(exist_ok=True)
subprocess.run(
    ["kaggle", "datasets", "download", "-d", "vatsalmavani/spotify-dataset", "-p", str(destination), "--unzip"],
    check=True,
)
print(destination / "data.csv")
