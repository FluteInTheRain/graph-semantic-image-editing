# Query analysis: clip_graph_flickr30k_lr1e4, seed 0, val

- alpha (chosen on val by training): 0.5
- explainer: occlusion on the trained model; top 3 relations per side
- selection rule: Only queries whose relevant image is in CLIP's top-k. change = CLIP rank - CLIP+Graph rank of the first relevant image. Success: change > 0, largest first; failure: change < 0, most negative first; ties: smaller query index first. A sentence already picked (same text, ignoring case and spaces) is skipped. 5 of each, fewer if not enough exist.
- queries: 25000; improved 7306, worsened 3557, unchanged 13288, relevant image not in candidates 849

A contribution is how much the final score drops when that one edge is removed
((1 - alpha) x drop of cos(q, h_i)): + means the edge pushed the image up, - means
it pulled the image down. The deletion check removes the top supporting
edges together and re-ranks the query's candidates. 'Without any edge' removes
every edge into the image: what is left is the image's own vector (W_self x_i + b).

## Success queries

### Query 11006: "(text not loaded; image 2966190737, text id 1)"

Source image 2966190737. First relevant image: CLIP rank 98 -> CLIP+Graph rank 5 (change +93).

**first relevant image: image 2966190737** (CLIP rank 98, CLIP+Graph rank 5; clip 0.2206, graph 0.3406, final 0.2806; 16 edges)

Pushed the image up:
- ballplayer.n.01 -[in]-> image 2966190737 (+0.0012)
- player.n.01 -[in]-> image 2966190737 (+0.0011)
- hit.n.01 -[in]-> image 2966190737 (+0.0006)
Pulled the image down:
- competition.n.01 -[in]-> image 2966190737 (-0.0009)
- wicket.n.04 -[in]-> image 2966190737 (-0.0005)
- semifinal.n.01 -[in]-> image 2966190737 (-0.0005)
Deletion check (remove 3 supporting edges): graph score 0.3406 -> 0.3341, CLIP+Graph rank 5 -> 5.
Without any edge (16 removed): graph score 0.3406 -> 0.2239, CLIP+Graph rank 5 -> 37.

### Query 10811: "(text not loaded; image 2925162156, text id 1)"

Source image 2925162156. First relevant image: CLIP rank 97 -> CLIP+Graph rank 8 (change +89).

**first relevant image: image 2925162156** (CLIP rank 97, CLIP+Graph rank 8; clip 0.2338, graph 0.2449, final 0.2393; 16 edges)

Pushed the image up:
- lap.n.01 -[in]-> image 2925162156 (+0.0008)
- pace.n.04 -[in]-> image 2925162156 (+0.0007)
- trave.n.01 -[in]-> image 2925162156 (+0.0006)
Pulled the image down:
- jogger.n.01 -[in]-> image 2925162156 (-0.0007)
- run.n.01 -[in]-> image 2925162156 (-0.0006)
- run.n.05 -[in]-> image 2925162156 (-0.0006)
Deletion check (remove 3 supporting edges): graph score 0.2449 -> 0.2399, CLIP+Graph rank 8 -> 8.
Without any edge (16 removed): graph score 0.2449 -> 0.1934, CLIP+Graph rank 8 -> 25.

### Query 15806: "(text not loaded; image 3826412077, text id 1)"

Source image 3826412077. First relevant image: CLIP rank 100 -> CLIP+Graph rank 11 (change +89).

**first relevant image: image 3826412077** (CLIP rank 100, CLIP+Graph rank 11; clip 0.1893, graph 0.2018, final 0.1956; 16 edges)

Pushed the image up:
- main.n.01 -[in]-> image 3826412077 (+0.0009)
- landmass.n.01 -[in]-> image 3826412077 (+0.0006)
- ma.n.01 -[in]-> image 3826412077 (+0.0006)
Pulled the image down:
- direction.n.01 -[in]-> image 3826412077 (-0.0005)
- direction.n.06 -[in]-> image 3826412077 (-0.0005)
- crew.n.01 -[in]-> image 3826412077 (-0.0005)
Deletion check (remove 3 supporting edges): graph score 0.2018 -> 0.1970, CLIP+Graph rank 11 -> 13.
Without any edge (16 removed): graph score 0.2018 -> 0.1910, CLIP+Graph rank 11 -> 15.

### Query 18359: "(text not loaded; image 4589015431, text id 4)"

Source image 4589015431. First relevant image: CLIP rank 99 -> CLIP+Graph rank 14 (change +85).

**first relevant image: image 4589015431** (CLIP rank 99, CLIP+Graph rank 14; clip 0.2555, graph 0.2717, final 0.2636; 16 edges)

Pushed the image up:
- trade_name.n.01 -[in]-> image 4589015431 (+0.0014)
- caption.n.01 -[in]-> image 4589015431 (+0.0011)
- commercial.n.01 -[in]-> image 4589015431 (+0.0011)
Pulled the image down:
- shopper.n.01 -[in]-> image 4589015431 (-0.0015)
- commuter.n.01 -[in]-> image 4589015431 (-0.0012)
- shopfront.n.01 -[in]-> image 4589015431 (-0.0006)
Deletion check (remove 3 supporting edges): graph score 0.2717 -> 0.2632, CLIP+Graph rank 14 -> 17.
Without any edge (16 removed): graph score 0.2717 -> 0.2106, CLIP+Graph rank 14 -> 44.

### Query 19459: "(text not loaded; image 4726413723, text id 4)"

Source image 4726413723. First relevant image: CLIP rank 94 -> CLIP+Graph rank 9 (change +85).

**first relevant image: image 4726413723** (CLIP rank 94, CLIP+Graph rank 9; clip 0.2659, graph 0.3152, final 0.2906; 16 edges)

Pushed the image up:
- main.n.01 -[in]-> image 4726413723 (+0.0015)
- strip.n.01 -[in]-> image 4726413723 (+0.0010)
- vicinity.n.01 -[in]-> image 4726413723 (+0.0008)
Pulled the image down:
- day.n.01 -[in]-> image 4726413723 (-0.0009)
- day.n.04 -[in]-> image 4726413723 (-0.0009)
- street.n.01 -[in]-> image 4726413723 (-0.0006)
Deletion check (remove 3 supporting edges): graph score 0.3152 -> 0.3075, CLIP+Graph rank 9 -> 10.
Without any edge (16 removed): graph score 0.3152 -> 0.2319, CLIP+Graph rank 9 -> 50.


## Failure queries

### Query 13974: "(text not loaded; image 3475245316, text id 4)"

Source image 3475245316. First relevant image: CLIP rank 13 -> CLIP+Graph rank 81 (change -68).

**first relevant image: image 3475245316** (CLIP rank 13, CLIP+Graph rank 81; clip 0.2458, graph 0.0960, final 0.1709; 16 edges)

Pushed the image up:
- passenger.n.01 -[in]-> image 3475245316 (+0.0015)
- waitress.n.01 -[in]-> image 3475245316 (+0.0014)
- onlooker.n.01 -[in]-> image 3475245316 (+0.0010)
Pulled the image down:
- sitting.n.01 -[in]-> image 3475245316 (-0.0023)
- footfall.n.01 -[in]-> image 3475245316 (-0.0010)
- spectator.n.01 -[in]-> image 3475245316 (-0.0006)
Deletion check (remove 3 supporting edges): graph score 0.0960 -> 0.0869, CLIP+Graph rank 81 -> 86.
Without any edge (16 removed): graph score 0.0960 -> 0.0821, CLIP+Graph rank 81 -> 89.

**rank 1 under CLIP+Graph: image 2314541227** (CLIP rank 3, CLIP+Graph rank 1; clip 0.2585, graph 0.2279, final 0.2432; 16 edges)

Pushed the image up:
- feather_boa.n.01 -[in]-> image 2314541227 (+0.0012)
- pat.n.01 -[in]-> image 2314541227 (+0.0011)
- waitress.n.01 -[in]-> image 2314541227 (+0.0007)
Pulled the image down:
- shopper.n.01 -[in]-> image 2314541227 (-0.0009)
- performer.n.01 -[in]-> image 2314541227 (-0.0008)
- wig.n.01 -[in]-> image 2314541227 (-0.0008)
Deletion check (remove 3 supporting edges): graph score 0.2279 -> 0.2209, CLIP+Graph rank 1 -> 1.
Without any edge (16 removed): graph score 0.2279 -> 0.1839, CLIP+Graph rank 1 -> 13.

### Query 15697: "(text not loaded; image 3793619579, text id 2)"

Source image 3793619579. First relevant image: CLIP rank 24 -> CLIP+Graph rank 83 (change -59).

**first relevant image: image 3793619579** (CLIP rank 24, CLIP+Graph rank 83; clip 0.2541, graph 0.1054, final 0.1797; 16 edges)

Pushed the image up:
- visitor.n.01 -[in]-> image 3793619579 (+0.0015)
- ball_boy.n.01 -[in]-> image 3793619579 (+0.0014)
- male_child.n.01 -[in]-> image 3793619579 (+0.0012)
Pulled the image down:
- seesaw.n.01 -[in]-> image 3793619579 (-0.0023)
- lap.n.01 -[in]-> image 3793619579 (-0.0011)
- fun.n.01 -[in]-> image 3793619579 (-0.0009)
Deletion check (remove 3 supporting edges): graph score 0.1054 -> 0.0959, CLIP+Graph rank 83 -> 88.
Without any edge (16 removed): graph score 0.1054 -> 0.0697, CLIP+Graph rank 83 -> 97.

**rank 1 under CLIP+Graph: image 4977290545** (CLIP rank 1, CLIP+Graph rank 1; clip 0.3233, graph 0.3153, final 0.3193; 16 edges)

Pushed the image up:
- knight.n.01 -[in]-> image 4977290545 (+0.0011)
- jockey.n.01 -[in]-> image 4977290545 (+0.0009)
- attendant.n.01 -[in]-> image 4977290545 (+0.0007)
Pulled the image down:
- hairnet.n.01 -[in]-> image 4977290545 (-0.0011)
- passerby.n.01 -[in]-> image 4977290545 (-0.0007)
- bystander.n.01 -[in]-> image 4977290545 (-0.0007)
Deletion check (remove 3 supporting edges): graph score 0.3153 -> 0.3092, CLIP+Graph rank 1 -> 1.
Without any edge (16 removed): graph score 0.3153 -> 0.2377, CLIP+Graph rank 1 -> 4.

### Query 17838: "(text not loaded; image 4521223019, text id 3)"

Source image 4521223019. First relevant image: CLIP rank 21 -> CLIP+Graph rank 78 (change -57).

**first relevant image: image 4521223019** (CLIP rank 21, CLIP+Graph rank 78; clip 0.2642, graph 0.1228, final 0.1935; 16 edges)

Pushed the image up:
- trave.n.01 -[in]-> image 4521223019 (+0.0011)
- helping.n.01 -[in]-> image 4521223019 (+0.0011)
- supply.n.01 -[in]-> image 4521223019 (+0.0009)
Pulled the image down:
- construction_worker.n.01 -[in]-> image 4521223019 (-0.0014)
- shopper.n.01 -[in]-> image 4521223019 (-0.0009)
- worker.n.01 -[in]-> image 4521223019 (-0.0006)
Deletion check (remove 3 supporting edges): graph score 0.1228 -> 0.1156, CLIP+Graph rank 78 -> 79.
Without any edge (16 removed): graph score 0.1228 -> 0.0909, CLIP+Graph rank 78 -> 90.

**rank 1 under CLIP+Graph: image 36026845** (CLIP rank 1, CLIP+Graph rank 1; clip 0.3206, graph 0.2939, final 0.3072; 16 edges)

Pushed the image up:
- ballast.n.01 -[in]-> image 36026845 (+0.0011)
- skid.n.01 -[in]-> image 36026845 (+0.0010)
- laying.n.01 -[in]-> image 36026845 (+0.0009)
Pulled the image down:
- stevedore.n.01 -[in]-> image 36026845 (-0.0023)
- construction_worker.n.01 -[in]-> image 36026845 (-0.0013)
- crew.n.01 -[in]-> image 36026845 (-0.0008)
Deletion check (remove 3 supporting edges): graph score 0.2939 -> 0.2870, CLIP+Graph rank 1 -> 1.
Without any edge (16 removed): graph score 0.2939 -> 0.2362, CLIP+Graph rank 1 -> 7.

### Query 5077: "(text not loaded; image 1065323785, text id 2)"

Source image 1065323785. First relevant image: CLIP rank 14 -> CLIP+Graph rank 70 (change -56).

**first relevant image: image 1065323785** (CLIP rank 14, CLIP+Graph rank 70; clip 0.2408, graph 0.1549, final 0.1979; 16 edges)

Pushed the image up:
- infrastructure.n.01 -[in]-> image 1065323785 (+0.0009)
- work.n.01 -[in]-> image 1065323785 (+0.0007)
- drive.n.01 -[in]-> image 1065323785 (+0.0005)
Pulled the image down:
- roadway.n.01 -[in]-> image 1065323785 (-0.0018)
- buggy.n.01 -[in]-> image 1065323785 (-0.0008)
- tarmacadam.n.01 -[in]-> image 1065323785 (-0.0008)
Deletion check (remove 3 supporting edges): graph score 0.1549 -> 0.1501, CLIP+Graph rank 70 -> 74.
Without any edge (16 removed): graph score 0.1549 -> 0.0919, CLIP+Graph rank 70 -> 95.

**rank 1 under CLIP+Graph: image 1066252238** (CLIP rank 1, CLIP+Graph rank 1; clip 0.3794, graph 0.4203, final 0.3998; 16 edges)

Pushed the image up:
- hit.n.01 -[in]-> image 1066252238 (+0.0019)
- playing.n.01 -[in]-> image 1066252238 (+0.0010)
- fun.n.01 -[in]-> image 1066252238 (+0.0010)
Pulled the image down:
- uphill.n.01 -[in]-> image 1066252238 (-0.0017)
- downhill.n.01 -[in]-> image 1066252238 (-0.0015)
- sawdust.n.01 -[in]-> image 1066252238 (-0.0007)
Deletion check (remove 3 supporting edges): graph score 0.4203 -> 0.4112, CLIP+Graph rank 1 -> 1.
Without any edge (16 removed): graph score 0.4203 -> 0.3208, CLIP+Graph rank 1 -> 1.

### Query 1544: "(text not loaded; image 166774567, text id 4)"

Source image 166774567. First relevant image: CLIP rank 8 -> CLIP+Graph rank 61 (change -53).

**first relevant image: image 166774567** (CLIP rank 8, CLIP+Graph rank 61; clip 0.2960, graph 0.1131, final 0.2045; 16 edges)

Pushed the image up:
- corner.n.01 -[in]-> image 166774567 (+0.0029)
- spectator.n.01 -[in]-> image 166774567 (+0.0010)
- runaway.n.01 -[in]-> image 166774567 (+0.0007)
Pulled the image down:
- footfall.n.01 -[in]-> image 166774567 (-0.0014)
- seller.n.01 -[in]-> image 166774567 (-0.0011)
- pedestrian.n.01 -[in]-> image 166774567 (-0.0009)
Deletion check (remove 3 supporting edges): graph score 0.1131 -> 0.1025, CLIP+Graph rank 61 -> 66.
Without any edge (16 removed): graph score 0.1131 -> 0.0761, CLIP+Graph rank 61 -> 84.

**rank 1 under CLIP+Graph: image 4807594495** (CLIP rank 1, CLIP+Graph rank 1; clip 0.3206, graph 0.2721, final 0.2964; 16 edges)

Pushed the image up:
- platform.n.01 -[in]-> image 4807594495 (+0.0018)
- block.n.01 -[in]-> image 4807594495 (+0.0016)
- plaza.n.01 -[in]-> image 4807594495 (+0.0012)
Pulled the image down:
- footfall.n.01 -[in]-> image 4807594495 (-0.0021)
- pedestrian.n.01 -[in]-> image 4807594495 (-0.0017)
- passerby.n.01 -[in]-> image 4807594495 (-0.0013)
Deletion check (remove 3 supporting edges): graph score 0.2721 -> 0.2615, CLIP+Graph rank 1 -> 2.
Without any edge (16 removed): graph score 0.2721 -> 0.1980, CLIP+Graph rank 1 -> 10.

