"""Connect the extraction steps: records -> embeddings -> saved, verified feature set."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from mmxai.data import flickr30k, visual_genome
from mmxai.data.records import ImageRecord, TextRecord
from mmxai.features.cache import compute_code_hash, feature_dir, save_features
from mmxai.features.verify import verify_features
from mmxai.utils.config import FLICKR30K, VISUAL_GENOME, FeatureConfig

# The modules whose content decides the output. Fixed order keeps the hash stable.
# Notebooks carry a copy of exactly these files (ADR-0041).
CODE_MODULES = [
    "utils/config.py",
    "data/records.py",
    "data/flickr30k.py",
    "data/visual_genome.py",
    "features/cache.py",
    "features/verify.py",
    "features/clip_encoder.py",
    "features/pipeline.py",
]

TEXT_SOURCES = {VISUAL_GENOME: "region_descriptions", FLICKR30K: "karpathy_captions"}
TEXT_ID_NAMES = {VISUAL_GENOME: "text_ids", FLICKR30K: "caption_index"}


def with_path_overrides(
    config: FeatureConfig,
    data_dir: str | None,
    images_dir: str | None,
    output_root: str | None,
) -> FeatureConfig:
    """Apply command line paths, which differ between a laptop and Kaggle."""
    if data_dir is not None:
        config = replace(config, data_dir=data_dir)
    if images_dir is not None:
        config = replace(config, images_dir=images_dir)
    if output_root is not None:
        config = replace(config, output_root=output_root)
    return config


def load_dataset_records(config: FeatureConfig) -> tuple[list[ImageRecord], list[TextRecord]]:
    """Return the images and texts to encode for the configured dataset."""
    data_dir = Path(config.data_dir)
    images_dir = None
    if config.images_dir is not None:
        images_dir = Path(config.images_dir)

    if config.dataset == VISUAL_GENOME:
        images = visual_genome.load_image_records(data_dir, images_dir)
        regions = visual_genome.load_region_texts(data_dir)
        texts = visual_genome.sample_regions(
            regions, config.regions_per_image, config.sampling_seed
        )
        return images, texts

    images = flickr30k.load_image_records(data_dir, images_dir)
    texts = flickr30k.load_captions(data_dir)
    return images, texts


def filter_texts_for_images(texts: list[TextRecord], kept_image_ids: set[int]) -> list[TextRecord]:
    """Keep only texts whose image exists, so every text has a valid retrieval target."""
    return [text for text in texts if text.image_id in kept_image_ids]


def apply_limits(
    images: list[ImageRecord],
    texts: list[TextRecord],
    limit_images: int | None,
    limit_texts: int | None,
) -> tuple[list[ImageRecord], list[TextRecord]]:
    """Cut the data down for a trial run."""
    if limit_images is not None:
        images = images[:limit_images]
        kept_image_ids = {image.image_id for image in images}
        texts = filter_texts_for_images(texts, kept_image_ids)
    if limit_texts is not None:
        texts = texts[:limit_texts]
    return images, texts


def check_failed_fraction(num_failed: int, num_total: int, max_fraction: float) -> None:
    """A few broken images are normal; many mean the download or the paths are wrong."""
    if num_failed > num_total * max_fraction:
        raise ValueError(
            f"{num_failed} of {num_total} images could not be read "
            f"(limit is {max_fraction:.2%}); check the download and images_dir"
        )


def build_meta(
    config: FeatureConfig,
    num_images: int,
    num_texts: int,
    embedding_dim: int,
    failed_image_ids: list[int],
    is_partial: bool,
    versions: dict[str, str],
    code_hash: str,
) -> dict:
    """Describe how a feature set was made, for checks and for the oral exam."""
    return {
        "dataset": config.dataset,
        "dataset_version": config.dataset_version,
        "model": config.model,
        "pretrained": config.pretrained,
        "embedding_dim": embedding_dim,
        "normalized": True,
        "num_images": num_images,
        "num_texts": num_texts,
        "text_source": TEXT_SOURCES[config.dataset],
        "regions_per_image": config.regions_per_image,
        "sampling_seed": config.sampling_seed,
        "failed_image_ids": failed_image_ids,
        "is_partial": is_partial,
        "open_clip_version": versions["open_clip"],
        "torch_version": versions["torch"],
        "code_hash": code_hash,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def run_extraction(
    config: FeatureConfig, limit_images: int | None = None, limit_texts: int | None = None
) -> Path:
    """Run one extraction and return the folder holding the verified result."""
    # Imported here because everything else in this module must work without torch.
    from mmxai.features import clip_encoder

    is_partial = limit_images is not None or limit_texts is not None
    output_dir = feature_dir(config, is_partial)
    # Checked first so a mistake costs seconds, not a whole encoding run.
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; delete it to extract again")

    images, texts = load_dataset_records(config)
    images, texts = apply_limits(images, texts, limit_images, limit_texts)
    print(f"{config.dataset}: {len(images)} images, {len(texts)} texts")

    bundle = clip_encoder.load_frozen_clip(config.model, config.pretrained, config.device)
    image_embeddings, image_ids, failed_ids = clip_encoder.encode_images(
        bundle, images, config.image_batch_size, config.num_workers
    )
    check_failed_fraction(len(failed_ids), len(images), config.max_failed_image_fraction)

    texts = filter_texts_for_images(texts, set(image_ids.tolist()))
    text_embeddings = clip_encoder.encode_texts(
        bundle, [text.text for text in texts], config.text_batch_size
    )

    package_root = Path(__file__).resolve().parent.parent
    meta = build_meta(
        config,
        num_images=len(image_ids),
        num_texts=len(texts),
        embedding_dim=image_embeddings.shape[1],
        failed_image_ids=failed_ids,
        is_partial=is_partial,
        versions=clip_encoder.library_versions(),
        code_hash=compute_code_hash(package_root, CODE_MODULES),
    )
    save_features(
        output_dir,
        image_embeddings=image_embeddings,
        image_ids=image_ids,
        text_embeddings=text_embeddings,
        text_image_ids=[text.image_id for text in texts],
        text_id_name=TEXT_ID_NAMES[config.dataset],
        text_id_values=[text.text_id for text in texts],
        meta=meta,
    )
    verify_features(output_dir)
    print(f"saved and verified: {output_dir}")
    return output_dir
