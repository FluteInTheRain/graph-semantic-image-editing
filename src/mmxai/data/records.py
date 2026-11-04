"""Plain records shared by every dataset loader, so the encoder needs no dataset knowledge."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageRecord:
    """One image to encode."""

    image_id: int
    path: Path


@dataclass(frozen=True)
class TextRecord:
    """One text to encode, linked to the image it describes."""

    image_id: int
    text_id: int  # region_id (Visual Genome) or caption_index (Flickr30k)
    text: str
