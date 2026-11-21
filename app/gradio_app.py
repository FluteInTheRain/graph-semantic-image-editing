"""Demo UI: type a query, pick a level, method and explanation, see the top-k val images.

Levels follow the assignment (PDF section 5):
    1  plain CLIP or CLIP + Graph (the team's one-layer GraphSAGE), basic explanation (paths)
    2  adds G-Retriever (PCST subgraph, stage 2)                         [not wired yet]
    3  adds the quantitative explanation (IG, m = 10, fidelity) for CLIP + Graph [not wired yet]

Only calls functions of src/mmxai/explain/ (query_path, path_view, path_graph); the UI
only decides which of them to call. Searches val only (ADR-0009). alpha is the one
training chose on val; the UI cannot change it.

Before running:
    uv sync --group ml --group graph --group demo
    The trained model: scripts/train_graph_rerank.py with the same --config and seed.
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
from mmxai.explain.path_graph import build_path_graph, draw_path_graph
from mmxai.explain.path_view import HOW_TO_READ, format_image_path
from mmxai.explain.query_path import (
    encode_query,
    load_engine,
    load_text_encoder,
    search_and_explain,
    search_clip,
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
NOT_WIRED = {
    G_RETRIEVER: "G-Retriever is not wired into the demo yet (next step).",
    QUANTITATIVE: "The quantitative explanation is not wired into the demo yet (next step).",
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
    args = parser.parse_args()

    config = load_graph_rerank_config(args.config)
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
        return gallery, note, None, "\n".join(rows)

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
        if method in NOT_WIRED:
            return [], NOT_WIRED[method], None, ""
        if explanation in NOT_WIRED:
            return [], NOT_WIRED[explanation], None, ""
        return answer_paths(text, k)

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
