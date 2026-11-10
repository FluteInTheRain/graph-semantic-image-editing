"""Download and unzip the Visual Genome annotation files the pipeline reads. Standard library only.

Each file is fetched once: an existing file is kept. A download goes to a temporary name
first, so an interrupted run never leaves a file that looks complete, and a file whose
sha256 differs from the recorded one is deleted instead of being used.
"""

import hashlib
import os
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from mmxai.utils.yaml_config import load_yaml_dataclass


@dataclass(frozen=True)
class DataConfig:
    """Where the files go and where they come from; see configs/data_vg.yaml."""

    data_dir: str
    files: list[dict]


@dataclass(frozen=True)
class RemoteFile:
    """One zipped JSON file: `name` is the file inside the zip and on disk."""

    name: str
    url: str
    sha256: str | None
    required: bool


def load_data_config(path: str | Path) -> tuple[Path, list[RemoteFile]]:
    config = load_yaml_dataclass(path, DataConfig)
    files = [RemoteFile(**entry) for entry in config.files]
    return Path(config.data_dir), files


def file_sha256(path: Path) -> str:
    """Hash in 1 MB chunks: objects.json is too large to read at once comfortably."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(remote: RemoteFile, data_dir: Path) -> Path:
    """Return data_dir/remote.name, downloading and unzipping it if it is missing."""
    target = data_dir / remote.name
    if target.exists():
        print(f"already there: {target}")
        return target
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / f"{remote.name}.zip.part"
    print(f"downloading {remote.url}")
    with urllib.request.urlopen(remote.url) as response, open(zip_path, "wb") as file:
        shutil.copyfileobj(response, file)

    unzipped = data_dir / f"{remote.name}.part"
    with zipfile.ZipFile(zip_path) as archive:
        if remote.name not in archive.namelist():
            zip_path.unlink()
            raise ValueError(f"{remote.url} does not contain {remote.name}")
        with archive.open(remote.name) as source, open(unzipped, "wb") as file:
            shutil.copyfileobj(source, file)
    zip_path.unlink()

    if remote.sha256 is not None:
        found = file_sha256(unzipped)
        if found != remote.sha256:
            unzipped.unlink()
            raise ValueError(
                f"{remote.name} has sha256 {found}, expected {remote.sha256} "
                "(data_cards/visual_genome.md); the file was deleted"
            )
    os.replace(unzipped, target)
    print(f"saved {target}")
    return target
