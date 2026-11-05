"""Copy src/mmxai modules and configs into the Kaggle notebook after a code change.

The notebook carries the code because Kaggle has no access to this repo (ADR-0041).
tests/test_notebooks_match_src.py fails if this script has not been run.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
NOTEBOOK = REPO_ROOT / "notebooks" / "10_prepare_clip_embeddings.ipynb"
COPY_ROOT = "/kaggle/temp/mmxai_src/"
MAGIC = "%%writefile "


def source_file_for(copy_path: str) -> Path:
    """Map a path inside the notebook's copy to the real file in the repo."""
    if copy_path.startswith("mmxai/"):
        return REPO_ROOT / "src" / copy_path
    return REPO_ROOT / copy_path


def sync_cell(cell: dict) -> bool:
    """Refresh one copied cell. Returns True if it changed."""
    text = "".join(cell["source"])
    if cell["cell_type"] != "code" or not text.startswith(MAGIC):
        return False

    first_line = text.split("\n", 1)[0]
    copy_path = first_line.removeprefix(MAGIC).strip().removeprefix(COPY_ROOT)
    new_text = first_line + "\n" + source_file_for(copy_path).read_text(encoding="utf-8")
    cell["source"] = new_text.splitlines(keepends=True)
    return new_text != text


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    changed = 0
    for cell in notebook["cells"]:
        if sync_cell(cell):
            changed += 1
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{changed} cells updated in {NOTEBOOK.name}")


if __name__ == "__main__":
    main()
