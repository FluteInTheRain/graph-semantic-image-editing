"""Demo UI: type a query, see the top-k val images and the paths behind each (ADR-0013).

Only calls functions of src/mmxai/explain/ (query_path, path_view); no logic of its own.
Searches val only (ADR-0009). alpha is the one training chose on val; the UI cannot change it.

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
)

MAX_K = 10  # more image nodes than this do not fit one screen during the oral exam
NO_IMAGES = (
    "No thumbnails: give --vg-data-dir with Visual Genome's image_data.json "
    "(images/ is optional; without it the original urls are shown, which needs internet)."
)


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

    def answer(text: str, k: int):
        results = search_and_explain(engine, encode_query(encoder, text), int(k))
        gallery = []
        for item in results:
            if item.image_id in sources:
                gallery.append((sources[item.image_id], f"#{item.rank}: image {item.image_id}"))
        note = "" if gallery else NO_IMAGES

        figure = draw_path_graph(build_path_graph(engine, results, args.num_edges), text)
        plt.close(figure)  # gradio keeps its own copy; without this every search leaks a figure

        tables = [HOW_TO_READ]
        for item in results:
            tables.append(format_image_path(engine, item, args.num_edges))
        return gallery, note, figure, "\n\n".join(tables)

    with gr.Blocks(title="CLIP + Graph: query paths") as demo:
        gr.Markdown(
            f"# CLIP + Graph on Visual Genome val\n"
            f"Experiment `{config.experiment}`, seed {args.seed}, alpha {engine.alpha:g} "
            f"(chosen on val by training). CLIP picks the top {engine.top_k}; "
            f"they are re-ranked by final = alpha × clip + (1 − alpha) × graph."
        )
        with gr.Row():
            query = gr.Textbox(label="Query", placeholder="man riding a horse", scale=4)
            k = gr.Slider(1, MAX_K, value=3, step=1, label="k")
        button = gr.Button("Search")
        gallery = gr.Gallery(label="Top-k images", columns=5, height="auto")
        note = gr.Markdown()
        graph = gr.Plot(label="Paths: query → concept → contains → image")
        with gr.Accordion("Numbers behind the graph", open=False):
            tables = gr.Markdown()
        outputs = [gallery, note, graph, tables]
        button.click(answer, inputs=[query, k], outputs=outputs)
        query.submit(answer, inputs=[query, k], outputs=outputs)

    demo.launch()


if __name__ == "__main__":
    main()
