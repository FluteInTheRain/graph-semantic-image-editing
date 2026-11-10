"""Downloading the Visual Genome annotation files, with file:// URLs instead of the network."""

import hashlib
import zipfile

import pytest

from mmxai.data.download import RemoteFile, download_file, load_data_config

CONTENT = b'[{"image_id": 1, "objects": []}]'


def make_zip(tmp_path, name="objects.json"):
    zip_path = tmp_path / "remote" / f"{name}.zip"
    zip_path.parent.mkdir()
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(name, CONTENT)
    return zip_path.as_uri()


def test_download_unzips_and_checks_the_hash(tmp_path):
    url = make_zip(tmp_path)
    remote = RemoteFile("objects.json", url, hashlib.sha256(CONTENT).hexdigest(), True)
    path = download_file(remote, tmp_path / "data")
    assert path.read_bytes() == CONTENT
    assert sorted(item.name for item in (tmp_path / "data").iterdir()) == ["objects.json"]


def test_wrong_hash_deletes_the_file(tmp_path):
    remote = RemoteFile("objects.json", make_zip(tmp_path), "0" * 64, True)
    with pytest.raises(ValueError, match="sha256"):
        download_file(remote, tmp_path / "data")
    assert list((tmp_path / "data").iterdir()) == []


def test_existing_file_is_not_downloaded_again(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "objects.json").write_bytes(b"kept")
    remote = RemoteFile("objects.json", "file:///does/not/exist.zip", None, True)
    assert download_file(remote, data_dir).read_bytes() == b"kept"


def test_zip_without_the_expected_file_is_rejected(tmp_path):
    remote = RemoteFile("objects.json", make_zip(tmp_path, name="other.json"), None, True)
    with pytest.raises(ValueError, match="does not contain"):
        download_file(remote, tmp_path / "data")


def test_committed_config_lists_objects_as_required():
    data_dir, files = load_data_config("configs/data_vg.yaml")
    required = {remote.name for remote in files if remote.required}
    assert required == {"objects.json"}
    assert str(data_dir) == "data/visual_genome"
