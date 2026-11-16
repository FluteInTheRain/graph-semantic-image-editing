"""Packed cache of per-text PCST subgraphs: packing, padding, save and load checks."""

import numpy as np
import pytest

from mmxai.graph.pcst_cache import load_cache, pack_subgraphs, padded_rows, save_cache, unpack_row


def test_pack_and_unpack():
    offsets, values = pack_subgraphs(
        [np.array([2, 5]), np.array([], dtype=np.int64), np.array([7])]
    )
    assert offsets.tolist() == [0, 2, 2, 3]
    assert unpack_row(offsets, values, 0).tolist() == [2, 5]
    assert unpack_row(offsets, values, 1).tolist() == []


def test_packing_nothing_gives_empty_arrays():
    offsets, values = pack_subgraphs([])
    assert offsets.tolist() == [0]
    assert values.tolist() == []
    assert values.dtype == np.int64


def test_padded_rows_use_minus_one():
    offsets, values = pack_subgraphs([np.array([2, 5]), np.array([7])])
    assert padded_rows(offsets, values).tolist() == [[2, 5], [7, -1]]


def test_padded_rows_of_all_empty_rows_have_width_one():
    offsets, values = pack_subgraphs([np.array([], dtype=np.int64)] * 2)
    assert padded_rows(offsets, values).tolist() == [[-1], [-1]]


def test_save_refuses_overwrite_and_load_checks_meta(tmp_path):
    offsets, values = pack_subgraphs([np.array([1])])
    meta = {"prize_k": 3, "edge_cost": 0.5, "num_texts": 1}
    save_cache(tmp_path, "val", offsets, values, meta)
    with pytest.raises(FileExistsError):
        save_cache(tmp_path, "val", offsets, values, meta)
    _, loaded_values = load_cache(tmp_path, "val", meta)
    assert loaded_values.tolist() == [1]
    with pytest.raises(ValueError, match="prize_k"):
        load_cache(tmp_path, "val", {"prize_k": 10, "edge_cost": 0.5, "num_texts": 1})


def test_load_checks_the_number_of_texts(tmp_path):
    offsets, values = pack_subgraphs([np.array([1])])
    save_cache(tmp_path, "val", offsets, values, {"prize_k": 3, "num_texts": 1})
    with pytest.raises(ValueError, match="num_texts"):
        load_cache(tmp_path, "val", {"prize_k": 3, "num_texts": 2})
