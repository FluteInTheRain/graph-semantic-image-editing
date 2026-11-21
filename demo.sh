#!/usr/bin/env bash
# Start the Gradio demo (app/gradio_app.py) with the team's default settings.
# Extra arguments are passed through and override the defaults, e.g.:
#   ./demo.sh --seed 1
#   ./demo.sh --vg-data-dir /path/to/visual_genome
set -euo pipefail

# Run from the repo root so the relative paths in configs/ resolve.
cd "$(dirname "$0")"

# The CLIP weights are already cached locally; offline mode stops the hub
# from hanging on a network check behind a proxy.
export HF_HUB_OFFLINE=1

exec uv run --system-certs --all-groups python app/gradio_app.py \
    --config configs/clip_graph_vg_lr1e4.yaml \
    --seed 0 \
    --vg-data-dir data/visual_genome \
    "$@"
