"""Evaluate retrieval on the verified query benchmark, direct and indirect reported apart.

The benchmark is evaluation-only (PDF section 6, ADR-0009): nothing here trains,
tunes or picks a threshold. It is read only after the configuration is fixed on val.

Any model plugs in through the QueryRanker protocol: given the query TEXTS and their
CLIP text embeddings, return the top_k image ids of the test gallery per query.
A ranker never receives the query records, so it cannot read the answers
(`relevant`) or the generator's evidence (`relation_evidence`, `path_hint`,
`hidden_names`): those exist for checking, not for scoring.
Plain CLIP is ClipIndexRanker; CLIP + Graph will be another class with the same
`rank` method, so both are scored by the same code on the same queries.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from mmxai.data.splits import load_split
from mmxai.eval.metrics import compute_metrics, first_relevant_rank
from mmxai.queries.schema import load_jsonl
from mmxai.queries.validator import validate_queries
from mmxai.utils.yaml_config import load_yaml_dataclass

QUERY_GROUPS = ("direct", "indirect")


@dataclass(frozen=True)
class QueryBenchmarkEvalConfig:
    """Settings of one benchmark evaluation; the gallery is always the test split."""

    experiment: str
    dataset: str
    split_file: str
    features_dir: str
    benchmark_path: str
    model: str
    pretrained: str
    device: str
    text_batch_size: int
    top_k: int
    recall_ks: list[int]
    output_dir: str


def load_query_benchmark_eval_config(path: str | Path) -> QueryBenchmarkEvalConfig:
    return load_yaml_dataclass(path, QueryBenchmarkEvalConfig)


class QueryRanker(Protocol):
    """Anything that ranks test images for queries (plain CLIP, CLIP + Graph, ...)."""

    def rank(self, texts: list[str], query_embeddings: np.ndarray, top_k: int) -> np.ndarray:
        """Return image ids of shape (num_queries, top_k), best first."""
        ...


class ClipIndexRanker:
    """Plain CLIP: exact cosine search in the FAISS index of the test images."""

    def __init__(self, image_index) -> None:
        self.image_index = image_index

    def rank(self, texts: list[str], query_embeddings: np.ndarray, top_k: int) -> np.ndarray:
        # Imported here so the metric code can be tested without faiss.
        from mmxai.index.image_index import search_images

        _, image_ids = search_images(self.image_index, query_embeddings, top_k)
        return image_ids


def load_verified_benchmark(path: str | Path, test_image_ids: set[str]) -> list[dict]:
    """Read the benchmark and refuse anything that is not a fully verified test benchmark."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. It is built only from human-verified pairs: generate "
            "candidates (scripts/generate_queries.py), have two members review the manifest, "
            "then run scripts/build_query_benchmark.py. See docs/query_benchmark.md."
        )
    records = load_jsonl(path)
    if not records:
        raise ValueError(f"{path} holds no queries")
    problems = validate_queries(records, test_image_ids, require_verified=True)
    if problems:
        raise ValueError(f"{path} is not a valid verified benchmark:\n" + "\n".join(problems))
    return records


def score_rankings(
    queries: list[dict], retrieved_ids: np.ndarray, recall_ks: list[int], top_k: int
) -> dict:
    """Metrics for all queries and for the direct and indirect groups, plus per-query ranks.

    Per-query ranks are kept so CLIP and CLIP + Graph can later be compared query by
    query (paired test), as ADR-0043 asks.
    """
    if len(queries) != len(retrieved_ids):
        raise ValueError(f"{len(queries)} queries but {len(retrieved_ids)} result lists")

    ranks_by_group: dict[str, list[int | None]] = {group: [] for group in QUERY_GROUPS}
    ranks_by_query_group: dict[str, dict[str, list[int | None]]] = {}
    all_ranks = []
    per_query = []
    for query, row in zip(queries, retrieved_ids, strict=True):
        relevant = {int(image_id) for image_id in query["relevant"]}
        rank = first_relevant_rank([int(image_id) for image_id in row], relevant)
        ranks_by_group[query["type"]].append(rank)
        all_ranks.append(rank)
        group = query.get("query_group")
        if group:
            ranks_by_query_group.setdefault(group, {"direct": [], "indirect": []})
            ranks_by_query_group[group][query["type"]].append(rank)
        per_query.append(
            {
                "qid": query["qid"],
                "pair_id": query.get("pair_id"),
                "type": query["type"],
                "query_group": query.get("query_group"),
                "rank": rank,
            }
        )

    metrics = {"all": compute_metrics(all_ranks, recall_ks, top_k)}
    for group in QUERY_GROUPS:
        # A group without queries is reported as absent instead of crashing or showing 0.
        if ranks_by_group[group]:
            metrics[group] = compute_metrics(ranks_by_group[group], recall_ks, top_k)
        else:
            metrics[group] = None
    # ADR-0050: activity and object queries are reported apart; v1 files have no groups.
    metrics["by_query_group"] = {}
    for group, ranks_by_type in sorted(ranks_by_query_group.items()):
        metrics["by_query_group"][group] = {}
        for query_type, ranks in ranks_by_type.items():
            if ranks:
                metrics["by_query_group"][group][query_type] = compute_metrics(
                    ranks, recall_ks, top_k
                )
            else:
                metrics["by_query_group"][group][query_type] = None
    metrics["per_query"] = per_query
    return metrics


def evaluate_benchmark(
    queries: list[dict],
    query_embeddings: np.ndarray,
    ranker: QueryRanker,
    recall_ks: list[int],
    top_k: int,
) -> dict:
    """Rank with any QueryRanker and score the result. The ranker only sees the texts."""
    texts = [query["text"] for query in queries]
    retrieved_ids = ranker.rank(texts, query_embeddings, top_k)
    return score_rankings(queries, retrieved_ids, recall_ks, top_k)


def encode_query_texts(config: QueryBenchmarkEvalConfig, texts: list[str]) -> np.ndarray:
    """Encode with the same frozen CLIP and tokenizer that produced the cached embeddings."""
    from mmxai.features.clip_encoder import encode_texts, load_frozen_clip

    bundle = load_frozen_clip(config.model, config.pretrained, config.device)
    return encode_texts(bundle, texts, config.text_batch_size)


def run_clip_benchmark_evaluation(config: QueryBenchmarkEvalConfig, seed: int) -> Path:
    """Evaluate plain CLIP on the verified benchmark and write a run folder (ADR-0010).

    The seed is only a label: frozen CLIP and an exact index give the same numbers
    for every seed (ADR-0011, ADR-0043).
    """
    # The benchmark is checked first: without verified queries nothing else matters.
    split = load_split(config.split_file)
    test_ids = split["test"]
    test_id_strings = {str(image_id) for image_id in test_ids.tolist()}
    queries = load_verified_benchmark(config.benchmark_path, test_id_strings)

    # Imported here so the metric code above can be tested without faiss and torch.
    import faiss

    from mmxai.features.cache import load_image_features, load_meta
    from mmxai.index.image_index import build_image_index
    from mmxai.index.vector_store import select_split_images
    from mmxai.utils.runs import collect_env, file_sha256, make_run_dir, write_run_files

    if not (Path(config.features_dir) / "meta.json").exists():
        raise FileNotFoundError(
            f"no CLIP features in {config.features_dir}; download them as described in "
            "docs/guides/chuan-bi-embedding-clip.md section 2"
        )
    meta = load_meta(config.features_dir)
    if meta["dataset"] != config.dataset or meta["is_partial"]:
        raise ValueError(f"{config.features_dir} is not a complete {config.dataset} feature set")
    # Query and image vectors must come from the same model, or cosine means nothing.
    if meta["model"] != config.model or meta["pretrained"] != config.pretrained:
        raise ValueError(
            f"features use {meta['model']}/{meta['pretrained']}, "
            f"config asks {config.model}/{config.pretrained}"
        )

    embeddings, image_ids = load_image_features(config.features_dir)
    test_embeddings, test_image_ids = select_split_images(embeddings, image_ids, test_ids)
    ranker = ClipIndexRanker(build_image_index(test_embeddings, test_image_ids))
    query_embeddings = encode_query_texts(config, [query["text"] for query in queries])

    metrics = evaluate_benchmark(queries, query_embeddings, ranker, config.recall_ks, config.top_k)
    metrics["split"] = "test"
    metrics["num_images"] = len(test_image_ids)
    metrics["split_file"] = config.split_file
    metrics["split_sha256"] = file_sha256(config.split_file)
    metrics["benchmark_sha256"] = file_sha256(config.benchmark_path)
    metrics["seed"] = seed
    metrics["features_code_hash"] = meta["code_hash"]

    run_dir = make_run_dir(config.output_dir, config.experiment, seed, datetime.now())
    env = collect_env({"numpy": np.__version__, "faiss": faiss.__version__})
    write_run_files(run_dir, asdict(config), metrics, env)
    return run_dir
