"""The demo shows a local image when it exists and the original url otherwise."""

from mmxai.data.image_sources import image_sources
from test_data_visual_genome import write_image_data


def test_image_sources_prefer_the_local_file_and_fall_back_to_the_url(tmp_path):
    write_image_data(tmp_path)
    local = tmp_path / "images" / "VG_100K" / "7.jpg"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"fake jpeg")

    sources = image_sources(tmp_path)

    assert sources[7] == str(local)
    assert sources[1] == "https://example.org/rak248/VG_100K_2/1.jpg"
