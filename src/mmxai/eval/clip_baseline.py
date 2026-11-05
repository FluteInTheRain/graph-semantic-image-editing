"""Evaluate plain CLIP text-to-image retrieval on one split (ADR-0043).

Steps: keep only the images and texts of the evaluated split, index the
images with FAISS, search with every text, and score the ranks.

Decision (ADR-0043): the gallery is every image of the evaluated split and the
queries are their sampled region descriptions, because the PDF (section 6) keeps
test images and test queries away from everything else.
Decision (ADR-0043): top_k = 100 (config), the same candidate count CLIP + Graph
will re-rank, so both models start from the same list.
Decision (ADR-0009): val may be run while developing; test is run once per seed,
on committed code, with nothing changed afterwards.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np

from mmxai.data.splits import load_split
from mmxai.eval.metrics import compute_metrics, first_relevant_rank
from mmxai.eval.relevance import same_text_relevant_images
from mmxai.features.cache import load_image_features, load_meta, load_text_features
from mmxai.index.image_index import build_image_index, search_images
from mmxai.utils.runs import collect_env, file_sha256, make_run_dir, write_run_files
from mmxai.utils.yaml_config import load_yaml_dataclass

# Decision (ADR-0009): never "train", because retrieval is only reported on held-out images.
EVAL_SPLITS = ("val", "test")


@dataclass(frozen=True)
class ClipEvalConfig:
    """Settings of one plain-CLIP evaluation."""

    experiment: str
    dataset: str
    split_file: str
    features_dir: str
    eval_split: str
    top_k: int
    recall_ks: list[int]
    same_text_threshold: float
    output_dir: str


def load_clip_eval_config(path: str | Path) -> ClipEvalConfig:
    """Load the YAML config and reject an evaluation split other than val or test."""
    config = load_yaml_dataclass(path, ClipEvalConfig)
    if config.eval_split not in EVAL_SPLITS:
        raise ValueError(f"eval_split must be one of {EVAL_SPLITS}, got {config.eval_split!r}")
    return config


def evaluate_clip(
    image_embeddings: np.ndarray,
    image_ids: np.ndarray,
    text_embeddings: np.ndarray,
    text_image_ids: np.ndarray,
    eval_image_ids: np.ndarray,
    top_k: int,
    recall_ks: list[int],
    same_text_threshold: float,
) -> dict:
    """Return the metrics of plain CLIP on the images listed in eval_image_ids."""
    missing = np.setdiff1d(eval_image_ids, image_ids)
    if len(missing) > 0:
        raise ValueError(f"{len(missing)} split images have no embedding, e.g. {missing[0]}")

    # Both the gallery and the queries come from the evaluated split only.
    image_rows = np.isin(image_ids, eval_image_ids)
    text_rows = np.isin(text_image_ids, eval_image_ids)
    gallery_embeddings = image_embeddings[image_rows]
    gallery_ids = image_ids[image_rows]
    query_embeddings = text_embeddings[text_rows]
    query_image_ids = text_image_ids[text_rows]

    index = build_image_index(gallery_embeddings, gallery_ids)
    _, retrieved_ids = search_images(index, query_embeddings, top_k)
    relevant = same_text_relevant_images(query_embeddings, query_image_ids, same_text_threshold)

    ranks = []
    for query in range(len(query_embeddings)):
        ranks.append(first_relevant_rank(retrieved_ids[query].tolist(), relevant[query]))

    metrics = compute_metrics(ranks, recall_ks, top_k)
    metrics["num_images"] = len(gallery_ids)
    several = sum(1 for images in relevant if len(images) > 1)
    metrics["fraction_queries_with_several_relevant"] = several / len(relevant)
    return metrics


def run_clip_evaluation(config: ClipEvalConfig, seed: int) -> Path:
    """Evaluate, write the run folder and return its path.

    Decision (ADR-0011): the seed is kept although it changes nothing (CLIP is
    frozen, the index is exact, so std over seeds is 0), so this run fits the same
    3-seed tables as CLIP + Graph.
    Decision (ADR-0009, ADR-0010): metrics.json records the split file hash and the
    feature code hash, so every number names exactly the data it came from.
    """
    meta = load_meta(config.features_dir)
    if meta["dataset"] != config.dataset or meta["is_partial"]:
        raise ValueError(f"{config.features_dir} is not a complete {config.dataset} feature set")

    image_embeddings, image_ids = load_image_features(config.features_dir)
    texts = load_text_features(config.features_dir)
    split = load_split(config.split_file)

    metrics = evaluate_clip(
        image_embeddings,
        image_ids,
        texts["embeddings"],
        texts["image_ids"],
        split[config.eval_split],
        config.top_k,
        config.recall_ks,
        config.same_text_threshold,
    )
    metrics["split"] = config.eval_split
    metrics["split_file"] = config.split_file
    metrics["split_sha256"] = file_sha256(config.split_file)
    metrics["seed"] = seed
    metrics["features_code_hash"] = meta["code_hash"]

    run_dir = make_run_dir(config.output_dir, config.experiment, seed, datetime.now())
    env = collect_env({"numpy": np.__version__, "faiss": faiss.__version__})
    write_run_files(run_dir, asdict(config), metrics, env)
    return run_dir
