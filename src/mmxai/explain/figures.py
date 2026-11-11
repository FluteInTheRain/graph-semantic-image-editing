"""Show one retrieval case as a row of images, for notebooks and slides.

Needs the Visual Genome images (folders VG_100K / VG_100K_2) and matplotlib + pillow;
without the images, notebooks still print every case as text.
"""

from pathlib import Path

from mmxai.explain.demo import QueryResult


def find_image(image_dirs: list[str | Path], image_id: int) -> Path | None:
    """Path of <image_id>.jpg in the first folder that has it, or None."""
    for folder in image_dirs:
        path = Path(folder) / f"{image_id}.jpg"
        if path.exists():
            return path
    return None


def show_case(result: QueryResult, image_dirs: list[str | Path], num_top: int = 3) -> None:
    """Draw the top images (green = correct, red = wrong) and, if not shown, the source image."""
    # Imported here so the rest of explain/ runs without matplotlib and pillow.
    import matplotlib.pyplot as plt
    from PIL import Image

    panels = []
    for rank, image_id in enumerate(result.top_image_ids[:num_top], start=1):
        correct = result.relevant_image_ids is not None and image_id in result.relevant_image_ids
        panels.append((image_id, f"rank {rank}", "green" if correct else "red"))
    shown = {image_id for image_id, _, _ in panels}
    if result.source_image_id is not None and result.source_image_id not in shown:
        rank = "not in top 100" if result.source_rank is None else f"rank {result.source_rank}"
        panels.append((result.source_image_id, f"source, {rank}", "green"))

    figure, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), squeeze=False)
    for axis, (image_id, label, color) in zip(axes[0], panels, strict=True):
        path = find_image(image_dirs, image_id)
        if path is None:
            axis.text(0.5, 0.5, "image not found", ha="center", va="center")
        else:
            axis.imshow(Image.open(path).convert("RGB"))
        axis.set_title(f"{label}\nimage {image_id}", color=color)
        axis.axis("off")
    case = "" if result.case is None else f"[{result.case}] "
    figure.suptitle(f"{case}{result.query}")
    figure.tight_layout()
    plt.show()
