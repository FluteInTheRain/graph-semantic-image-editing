"""Evaluate plain CLIP text-to-image retrieval on one split (ADR-0043).

Steps: keep only the images and texts of the evaluated split, index the
images with FAISS, search with every text, and score the ranks.

Decision (ADR-0043): queries are the sampled region descriptions of the evaluated
split's images; the gallery is every image of that split.
  Why: the PDF (section 6) keeps test images and test queries away from everything else.
  Decided by: FluteInTheRain, 2026-11-05 (proposed by Claude Code).

Decision (ADR-0043): retrieve top_k = 100 candidates (config).
  Why: the same candidate list CLIP + Graph will re-rank, so both models start equal.
  Decided by: FluteInTheRain, template configs/clip_graph_vg.yaml; kept in ADR-0043.

Decision (ADR-0009): develop on val; run test once per seed on committed code and
change nothing afterwards.
  Why: choosing anything after seeing test numbers would leak test into the model.
  Decided by: FluteInTheRain, ADR-0009. Done this way on 2026-11-05 (commit c1678ef).
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np

from mmxai.data.splits import load_split
from mmxai.eval.extended import (
    check_extension_settings,
    image_to_text_report,
    query_groupings,
    text_to_image_report,
)
from mmxai.eval.image_to_text import retrieve_text_candidates
from mmxai.eval.metrics import compute_metrics, first_relevant_rank
from mmxai.eval.query_groups import Predicates, load_predicates
from mmxai.features.cache import load_image_features, load_meta, load_text_features
from mmxai.rerank.fusion import retrieve_candidates
from mmxai.utils.runs import (
    collect_env,
    file_sha256,
    make_run_dir,
    test_run_exists,
    write_run_files,
)
from mmxai.utils.yaml_config import load_yaml_dataclass

# Decision (ADR-0009): never evaluate on "train".
#   Why: retrieval is only reported on held-out images.
#   Decided by: FluteInTheRain, ADR-0009.
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
    # Optional extensions (spec 2026-11-13). All None = the original evaluation, unchanged.
    # Plain None defaults, not default_factory: load_yaml_dataclass treats those as required.
    ndcg_k: int | None = None
    image_to_text: bool | None = None
    vg_data_dir: str | None = None
    predicates_file: str | None = None
    extra_action_predicates: list[str] | None = None
    extra_spatial_predicates: list[str] | None = None
    excluded_predicates: list[str] | None = None


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
    ndcg_k: int | None = None,
    image_to_text: bool = False,
    group_inputs: tuple[np.ndarray, str, Predicates] | None = None,
) -> dict:
    """Return the metrics of plain CLIP on the images listed in eval_image_ids.

    ndcg_k, image_to_text and group_inputs switch on the extended report. group_inputs is
    (query text ids, Visual Genome folder, predicates). With all three off the result is
    the original one.
    """
    # The same function CLIP + Graph uses, so both re-rank the same candidates.
    candidates = retrieve_candidates(
        image_embeddings,
        image_ids,
        text_embeddings,
        text_image_ids,
        eval_image_ids,
        top_k,
        same_text_threshold,
    )

    if ndcg_k is None:
        ranks = []
        for query in range(len(candidates.candidate_ids)):
            retrieved = candidates.candidate_ids[query].tolist()
            ranks.append(first_relevant_rank(retrieved, candidates.relevant[query]))
        metrics = compute_metrics(ranks, recall_ks, top_k)
    else:
        groupings = None
        if group_inputs is not None:
            query_text_ids, vg_data_dir, predicates = group_inputs
            groupings = query_groupings(candidates, query_text_ids, vg_data_dir, predicates)
        # Plain CLIP: the final score is the CLIP score itself.
        metrics = text_to_image_report(
            candidates, candidates.clip_scores, groupings, recall_ks, top_k, ndcg_k
        )
    metrics["num_images"] = int(np.isin(image_ids, eval_image_ids).sum())
    several = sum(1 for images in candidates.relevant if len(images) > 1)
    metrics["fraction_queries_with_several_relevant"] = several / len(candidates.relevant)

    if image_to_text:
        text_candidates = retrieve_text_candidates(
            image_embeddings,
            image_ids,
            text_embeddings,
            text_image_ids,
            eval_image_ids,
            top_k,
            same_text_threshold,
        )
        metrics["image_to_text"] = image_to_text_report(
            text_candidates, text_candidates.clip_scores, recall_ks, top_k, ndcg_k
        )
        # alpha = 1 means the pair score is CLIP only.
        metrics["image_to_text"]["alpha"] = 1.0
    return metrics


def run_clip_evaluation(config: ClipEvalConfig, seed: int) -> Path:
    """Evaluate, write the run folder and return its path.

    Decision (ADR-0011): keep a seed argument although it changes nothing here.
      Why: CLIP is frozen and the index is exact (std over seeds is 0), but the run
      must fit the same 3-seed tables as CLIP + Graph.
      Decided by: FluteInTheRain, ADR-0011 (applied by Claude Code, 2026-11-05).

    Decision (ADR-0009, ADR-0010): metrics.json records the split file hash and the
    feature code hash.
      Why: every number must name exactly the data it came from.
      Decided by: FluteInTheRain, ADR-0009 and ADR-0010.
    """
    # Decision (ADR-0009): refuse a second test run of the same experiment and seed.
    #   Why: test runs once with the locked config; a repeat means test could steer choices.
    #   Decided by: FluteInTheRain, ADR-0009; this check proposed by Claude Code, 2026-11-13.
    if config.eval_split == "test" and test_run_exists(config.output_dir, config.experiment, seed):
        raise FileExistsError(f"test of {config.experiment} seed {seed} was already run")
    check_extension_settings(
        config.ndcg_k, config.image_to_text, config.vg_data_dir, config.predicates_file
    )

    meta = load_meta(config.features_dir)
    if meta["dataset"] != config.dataset or meta["is_partial"]:
        raise ValueError(f"{config.features_dir} is not a complete {config.dataset} feature set")

    image_embeddings, image_ids = load_image_features(config.features_dir)
    texts = load_text_features(config.features_dir)
    split = load_split(config.split_file)

    group_inputs = None
    if config.vg_data_dir is not None:
        predicates = load_predicates(
            config.predicates_file,
            config.extra_action_predicates,
            config.extra_spatial_predicates,
            config.excluded_predicates,
        )
        # The same mask retrieve_candidates uses, so ids line up with the queries.
        query_rows = np.isin(texts["image_ids"], split[config.eval_split])
        group_inputs = (texts["text_ids"][query_rows], config.vg_data_dir, predicates)

    metrics = evaluate_clip(
        image_embeddings,
        image_ids,
        texts["embeddings"],
        texts["image_ids"],
        split[config.eval_split],
        config.top_k,
        config.recall_ks,
        config.same_text_threshold,
        config.ndcg_k,
        bool(config.image_to_text),
        group_inputs,
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
