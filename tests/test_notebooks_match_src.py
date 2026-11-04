"""The Kaggle notebook carries a copy of the code; these tests keep that copy honest."""

import json
import re
from pathlib import Path

from mmxai.features.pipeline import CODE_MODULES

REPO_ROOT = Path(__file__).parent.parent
NOTEBOOK = REPO_ROOT / "notebooks" / "10_prepare_clip_embeddings.ipynb"
COPY_ROOT = "/kaggle/temp/mmxai_src"
COPIED_CONFIGS = ["configs/features_visual_genome.yaml", "configs/features_flickr30k.yaml"]
EXPECTED_SECTIONS = ["Config", "Code", "Smoke", "Dataset 1", "Dataset 2"]


def load_cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def cell_text(cell: dict) -> str:
    return "".join(cell["source"])


def copied_cells() -> dict[str, str]:
    """Return {path below COPY_ROOT: copied text} for every %%writefile cell."""
    copies = {}
    for cell in load_cells():
        text = cell_text(cell)
        if cell["cell_type"] == "code" and text.startswith("%%writefile "):
            first_line, body = text.split("\n", 1)
            path = first_line.removeprefix("%%writefile ").strip()
            assert path.startswith(COPY_ROOT + "/")
            copies[path.removeprefix(COPY_ROOT + "/")] = body
    return copies


def test_notebook_cells_match_src():
    copies = copied_cells()
    for module in CODE_MODULES:
        source = (REPO_ROOT / "src" / "mmxai" / module).read_text(encoding="utf-8")
        assert copies[f"mmxai/{module}"] == source, f"run scripts/sync_notebook_code.py ({module})"


def test_notebook_configs_match_repo():
    copies = copied_cells()
    for config in COPIED_CONFIGS:
        source = (REPO_ROOT / config).read_text(encoding="utf-8")
        assert copies[config] == source, f"run scripts/sync_notebook_code.py ({config})"


def test_notebook_contains_every_code_module():
    expected = {f"mmxai/{module}" for module in CODE_MODULES} | set(COPIED_CONFIGS)
    assert set(copied_cells()) == expected


def test_notebook_sections_in_order():
    headings = []
    for cell in load_cells():
        if cell["cell_type"] == "markdown" and cell_text(cell).startswith("## "):
            headings.append(cell_text(cell).splitlines()[0])
    positions = []
    for section in EXPECTED_SECTIONS:
        matching = [index for index, heading in enumerate(headings) if section in heading]
        assert len(matching) == 1, section
        positions.append(matching[0])
    assert positions == sorted(positions)


def test_only_one_dataset_setting():
    code_cells = [cell_text(cell) for cell in load_cells() if cell["cell_type"] == "code"]
    assignment = re.compile(r"^DATASET\s*=", re.MULTILINE)
    assert assignment.search(code_cells[0])
    assert sum(len(assignment.findall(text)) for text in code_cells) == 1


def test_smoke_output_stays_out_of_published_folder():
    text = "\n".join(cell_text(cell) for cell in load_cells() if cell["cell_type"] == "code")
    assert re.search(r'SMOKE_OUTPUT_ROOT\s*=\s*"/kaggle/temp/', text)
    assert re.search(r'OUTPUT_ROOT\s*=\s*"/kaggle/working/', text)


def test_notebook_has_no_outputs():
    for cell in load_cells():
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
