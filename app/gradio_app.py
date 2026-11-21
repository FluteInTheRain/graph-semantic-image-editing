"""Demo UI: type a query, pick a level, method and explanation, see the top-k val images.

Levels follow the assignment (PDF section 5):
    1  plain CLIP or CLIP + Graph (the team's one-layer GraphSAGE), basic explanation (paths)
    2  adds G-Retriever stage 2 (GraphSAGE on the query's PCST subgraph), explained by
       the concepts its score uses
    3  adds the quantitative explanation of CLIP + Graph: the main explainer and m chosen
       on val (configs/explain_vg.yaml), with fidelity, sparsity and validity

Only calls functions of src/mmxai/explain/; the UI only decides which of them to call.
Searches val only (ADR-0009). Every alpha is the one training chose on val; the UI
cannot change it.

Before running:
    uv sync --group ml --group graph --group demo
    The trained model: scripts/train_graph_rerank.py with the same --config and seed.
    For G-Retriever (loaded the first time it is picked): the PMI concept edges and the
    stage-2 model of --retriever-config (scripts/train_pcst_sage.py).
    Optional, for thumbnails: Visual Genome image_data.json and the images/ folder.

Run (from the repo root; loading takes about 10 s on a laptop CPU):
    uv run python app/gradio_app.py --config configs/clip_graph_vg_lr1e4.yaml --seed 0 \
        --vg-data-dir data/visual_genome
"""

import argparse
from pathlib import Path

import gradio as gr
from matplotlib import pyplot as plt

from mmxai.data.image_sources import image_sources
from mmxai.data.visual_genome import IMAGE_FILE
from mmxai.eval.graph_rerank import load_graph_rerank_config
from mmxai.eval.pcst_sage_rerank import load_pcst_sage_config
from mmxai.explain.config import load_explain_config
from mmxai.explain.evaluation import check_settings
from mmxai.explain.path_graph import build_path_graph, draw_path_graph
from mmxai.explain.path_view import HOW_TO_READ, format_image_path
from mmxai.explain.quantitative_view import (
    HOW_TO_READ_QUANTITATIVE,
    format_quantified,
    format_quantified_summary,
)
from mmxai.explain.query_path import (
    encode_query,
    load_engine,
    load_text_encoder,
    search_and_explain,
    search_clip,
)
from mmxai.explain.query_quantitative import explain_quantitatively
from mmxai.explain.retriever_path import load_retriever_engine, search_retriever
from mmxai.explain.retriever_view import (
    HOW_TO_READ_RETRIEVER,
    format_retriever_result,
    format_subgraph,
)

MAX_K = 10  # more image nodes than this do not fit one screen during the oral exam
NO_IMAGES = (
    "No thumbnails: give --vg-data-dir with Visual Genome's image_data.json "
    "(images/ is optional; without it the original urls are shown, which needs internet)."
)

LEVELS = ["1", "2", "3"]
CLIP_ONLY = "Plain CLIP"
CLIP_GRAPH = "CLIP + Graph (GraphSAGE, ours)"
G_RETRIEVER = "G-Retriever (PCST, stage 2)"
BASIC = "Basic: paths (Occlusion)"
QUANTITATIVE = "Quantitative: IG (m = 10) + fidelity"

# What each level of the assignment adds (PDF section 5).
METHODS_BY_LEVEL = {
    "1": [CLIP_ONLY, CLIP_GRAPH],
    "2": [CLIP_ONLY, CLIP_GRAPH, G_RETRIEVER],
    "3": [CLIP_ONLY, CLIP_GRAPH, G_RETRIEVER],
}


def explanations_for(level: str, method: str) -> list[str]:
    """Explanations the UI offers for this level and method.

    Plain CLIP has no graph, so nothing to explain. The quantitative explanation is level 3
    and, as in the report, only for the team's model (the explainers run on its GraphSAGE).
    """
    if method == CLIP_ONLY:
        return []
    if level == "3" and method == CLIP_GRAPH:
        return [BASIC, QUANTITATIVE]
    return [BASIC]


def load_sources(vg_data_dir: str | None) -> dict[int, str]:
    """image id -> local file or url; empty when image_data.json is missing."""
    if vg_data_dir is None or not (Path(vg_data_dir) / IMAGE_FILE).exists():
        return {}
    return image_sources(Path(vg_data_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-edges", type=int, default=3, help="edges shown per side")
    parser.add_argument("--vg-data-dir", help="folder with image_data.json (and images/)")
    parser.add_argument(
        "--explain-config",
        default="configs/explain_vg.yaml",
        help="level 3: main explainer and m chosen on val for this model",
    )
    parser.add_argument(
        "--retriever-config",
        default="configs/clip_pcst_sage_vg.yaml",
        help="level 2: the G-Retriever stage-2 experiment",
    )
    args = parser.parse_args()

    config = load_graph_rerank_config(args.config)
    explain_config = load_explain_config(args.explain_config)
    # The explainer and m were chosen on val for one model; never use them on another.
    quantitative_problem = None
    if Path(explain_config.ranking_config) != Path(args.config):
        quantitative_problem = (
            f"The quantitative explanation is not available: {args.explain_config} was "
            f"chosen on val for {explain_config.ranking_config}, not for {args.config}."
        )
    else:
        check_settings(explain_config, config)
    engine = load_engine(config, args.seed)
    encoder = load_text_encoder(config)
    sources = load_sources(args.vg_data_dir)
    print(f"{len(sources)} image sources found (thumbnails)")

    def thumbnails(ranked: list[tuple[int, int]]) -> list[tuple[str, str]]:
        """(rank, image id) pairs -> gallery items for the images we have a source for."""
        gallery = []
        for rank, image_id in ranked:
            if image_id in sources:
                gallery.append((sources[image_id], f"#{rank}: image {image_id}"))
        return gallery

    def answer_clip(text: str, k: int):
        hits = search_clip(engine, encode_query(encoder, text), k)
        gallery = thumbnails([(hit.rank, hit.image_id) for hit in hits])
        rows = ["| rank | image | CLIP cosine |", "|---|---|---|"]
        for hit in hits:
            rows.append(f"| {hit.rank} | {hit.image_id} | {hit.clip_score:.4f} |")
        note = "Plain CLIP ranks by cosine only: no graph, so no explanation."
        if not gallery:
            note += "\n\n" + NO_IMAGES
        return gallery, note, gr.Plot(value=None, visible=False), "\n".join(rows)

    retriever = {}  # loaded on first use: it reads every feature again (about a minute)

    def get_retriever():
        if "engine" not in retriever:
            retriever_config = load_pcst_sage_config(args.retriever_config)
            retriever["engine"] = load_retriever_engine(retriever_config, args.seed)
        return retriever["engine"]

    def answer_retriever(text: str, k: int):
        try:
            retriever_engine = get_retriever()
        except FileNotFoundError as error:
            message = (
                f"G-Retriever cannot be loaded: {error}. It needs the PMI concept edges "
                f"and the stage-2 model of {args.retriever_config} (see app/README.md)."
            )
            return [], message, None, ""
        answer = search_retriever(retriever_engine, encode_query(encoder, text), k)
        gallery = thumbnails([(item.rank, item.image_id) for item in answer.results])
        alpha = retriever_engine.alpha
        note = (
            f"G-Retriever stage 2, alpha {alpha:g} (chosen on val by training).\n\n"
            + format_subgraph(answer)
        )
        if not gallery:
            note += "\n\n" + NO_IMAGES
        tables = [HOW_TO_READ_RETRIEVER]
        for item in answer.results:
            tables.append(format_retriever_result(item, retriever_engine.alpha))
        return gallery, note, None, "\n\n".join(tables)

    def answer_quantitative(text: str, k: int):
        if quantitative_problem is not None:
            return [], quantitative_problem, None, ""
        items = explain_quantitatively(
            engine, encode_query(encoder, text), k, explain_config, args.seed
        )
        gallery = thumbnails([(item.rank, item.image_id) for item in items])
        explainer = explain_config.main_explainer
        note = (
            f"Explainer {explainer}, m = {explain_config.top_m} (both chosen on val).\n\n"
            + format_quantified_summary(items)
        )
        if not gallery:
            note += "\n\n" + NO_IMAGES
        tables = [HOW_TO_READ_QUANTITATIVE]
        for item in items:
            tables.append(format_quantified(engine, item, explainer))
        return gallery, note, None, "\n\n".join(tables)

    def answer_paths(text: str, k: int):
        results = search_and_explain(engine, encode_query(encoder, text), k)
        gallery = thumbnails([(item.rank, item.image_id) for item in results])
        note = "" if gallery else NO_IMAGES

        figure = draw_path_graph(build_path_graph(engine, results, args.num_edges), text)
        plt.close(figure)  # gradio keeps its own copy; without this every search leaks a figure

        tables = [HOW_TO_READ]
        for item in results:
            tables.append(format_image_path(engine, item, args.num_edges))
        return gallery, note, figure, "\n\n".join(tables)

    def answer(text: str, k: int, method: str, explanation: str | None):
        k = int(k)
        if method == CLIP_ONLY:
            return answer_clip(text, k)
        if method == G_RETRIEVER:
            gallery, note, figure, tables = answer_retriever(text, k)
        elif explanation == QUANTITATIVE:
            gallery, note, figure, tables = answer_quantitative(text, k)
        else:
            gallery, note, figure, tables = answer_paths(text, k)
        # The path graph exists only for the basic CLIP + Graph explanation.
        return gallery, note, gr.Plot(value=figure, visible=figure is not None), tables

    def on_level(level: str, method: str):
        methods = METHODS_BY_LEVEL[level]
        if method not in methods:
            method = CLIP_GRAPH
        return gr.Radio(choices=methods, value=method), explanation_radio(level, method)

    def explanation_radio(level: str, method: str):
        choices = explanations_for(level, method)
        value = choices[0] if choices else None
        return gr.Radio(choices=choices, value=value, visible=bool(choices))

    with gr.Blocks(title="CLIP + Graph: query paths") as demo:
        gr.Markdown(
            f"# Multimodal retrieval with graph explanations (Visual Genome val)\n"
            f"Experiment `{config.experiment}`, seed {args.seed}, alpha {engine.alpha:g} "
            f"(chosen on val by training). CLIP picks the top {engine.top_k}; "
            f"CLIP + Graph re-ranks them by final = alpha × clip + (1 − alpha) × graph."
        )
        with gr.Row():
            level = gr.Radio(LEVELS, value="1", label="Level")
            method = gr.Radio(METHODS_BY_LEVEL["1"], value=CLIP_GRAPH, label="Method")
            explanation = gr.Radio(
                explanations_for("1", CLIP_GRAPH), value=BASIC, label="Explanation"
            )
        with gr.Row():
            query = gr.Textbox(label="Query", placeholder="man riding a horse", scale=4)
            k = gr.Slider(1, MAX_K, value=3, step=1, label="k")
        button = gr.Button("Search")
        gallery = gr.Gallery(label="Top-k images", columns=5, height="auto")
        note = gr.Markdown()
        graph = gr.Plot(label="Paths: query → concept → contains → image")
        with gr.Accordion("Numbers behind the result", open=False):
            tables = gr.Markdown()

        level.change(on_level, inputs=[level, method], outputs=[method, explanation])
        method.change(explanation_radio, inputs=[level, method], outputs=explanation)
        inputs = [query, k, method, explanation]
        outputs = [gallery, note, graph, tables]
        button.click(answer, inputs=inputs, outputs=outputs)
        query.submit(answer, inputs=inputs, outputs=outputs)

    demo.launch()


if __name__ == "__main__":
    main()
