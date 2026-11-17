# Explanation evaluation: clip_graph_vg_lr1e4, test

Seeds [0, 1, 2], 20 queries, explained image: rank 1 under CLIP + Graph. Mean ± std over seeds.

| explainer | m | Fid+ | random Fid+ | rank drop | random rank drop | Fid- | characterization | robust Fid+ | sparsity | p (Fid+ > random) | p (rank > random) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| integrated_gradients | 10 | 0.0225 ± 0.0010 | 0.0006 ± 0.0002 | 2.5667 ± 0.2930 | 0.2558 ± 0.0816 | -0.0124 ± 0.0004 | 0.5043 ± 0.0362 | 0.0010 ± 0.0001 | 0.3750 ± 0.0000 | 9.54e-07 | 0.000488 |
