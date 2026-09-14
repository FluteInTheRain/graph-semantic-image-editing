"""Smoke test: the package and every pipeline subpackage import without heavy deps."""

import importlib

SUBPACKAGES = ["data", "features", "index", "graph", "models", "rerank", "explain", "eval", "utils"]


def test_import_mmxai():
    import mmxai

    assert mmxai.__version__
    for name in SUBPACKAGES:
        assert importlib.import_module(f"mmxai.{name}").__doc__
