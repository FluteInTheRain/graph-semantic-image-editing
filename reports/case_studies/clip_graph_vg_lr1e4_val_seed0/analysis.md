# Query analysis: clip_graph_vg_lr1e4, seed 0, val

- alpha (chosen on val by training): 0.0
- explainer: occlusion on the trained model; top 3 relations per side
- selection rule: Only queries whose relevant image is in CLIP's top-k. change = CLIP rank - CLIP+Graph rank of the first relevant image. Success: change > 0, largest first; failure: change < 0, most negative first; ties: smaller query index first. A sentence already picked (same text, ignoring case and spaces) is skipped. 5 of each, fewer if not enough exist.
- queries: 25000; improved 7387, worsened 4620, unchanged 1463, relevant image not in candidates 11530

A contribution is how much the final score drops when that one edge is removed
((1 - alpha) x drop of cos(q, h_i)): + means the edge pushed the image up, - means
it pulled the image down. The deletion check removes the top supporting
edges together and re-ranks the query's candidates. 'Without any edge' removes
every edge into the image: what is left is the image's own vector (W_self x_i + b).

## Success queries

### Query 20811: "man in a grey shirt and blue jeans"

Source image 2398911. First relevant image: CLIP rank 99 -> CLIP+Graph rank 1 (change +98).

**first relevant image: image 2398911** (CLIP rank 99, CLIP+Graph rank 1; clip 0.1811, graph 0.2046, final 0.2046; 16 edges)

Pushed the image up:
- competition.n.01 -[in]-> image 2398911 (+0.0030)
- action.n.01 -[in]-> image 2398911 (+0.0025)
- scene.n.01 -[in]-> image 2398911 (+0.0017)
Pulled the image down:
- tennis.n.01 -[in]-> image 2398911 (-0.0027)
- seesaw.n.01 -[in]-> image 2398911 (-0.0021)
- trick.n.01 -[in]-> image 2398911 (-0.0017)
Deletion check (remove 3 supporting edges): graph score 0.2046 -> 0.1962, CLIP+Graph rank 1 -> 1.
Without any edge (16 removed): graph score 0.2046 -> 0.1679, CLIP+Graph rank 1 -> 7.

### Query 4623: "Blue jacket with orange stripes"

Source image 2326205. First relevant image: CLIP rank 100 -> CLIP+Graph rank 3 (change +97).

**first relevant image: image 2326205** (CLIP rank 100, CLIP+Graph rank 3; clip 0.1685, graph 0.1771, final 0.1771; 16 edges)

Pushed the image up:
- ski_run.n.01 -[in]-> image 2326205 (+0.0017)
- ski_resort.n.01 -[in]-> image 2326205 (+0.0016)
- downhill.n.01 -[in]-> image 2326205 (+0.0014)
Pulled the image down:
- ski_cap.n.01 -[in]-> image 2326205 (-0.0041)
- ski_pole.n.01 -[in]-> image 2326205 (-0.0018)
- snowshoe.n.01 -[in]-> image 2326205 (-0.0016)
Deletion check (remove 3 supporting edges): graph score 0.1771 -> 0.1717, CLIP+Graph rank 3 -> 5.
Without any edge (16 removed): graph score 0.1771 -> 0.1658, CLIP+Graph rank 3 -> 5.

### Query 4715: "part of a trouser"

Source image 2326733. First relevant image: CLIP rank 100 -> CLIP+Graph rank 3 (change +97).

**first relevant image: image 2387245** (CLIP rank 100, CLIP+Graph rank 3; clip 0.1743, graph 0.0727, final 0.0727; 16 edges)

Pushed the image up:
- catcher.n.01 -[in]-> image 2387245 (+0.0014)
- player.n.01 -[in]-> image 2387245 (+0.0009)
- outfielder.n.01 -[in]-> image 2387245 (+0.0007)
Pulled the image down:
- pitcher.n.01 -[in]-> image 2387245 (-0.0009)
- one-third.n.01 -[in]-> image 2387245 (-0.0008)
- pitch.n.01 -[in]-> image 2387245 (-0.0006)
Deletion check (remove 3 supporting edges): graph score 0.0727 -> 0.0693, CLIP+Graph rank 3 -> 4.
Without any edge (16 removed): graph score 0.0727 -> 0.0987, CLIP+Graph rank 3 -> 1.

### Query 7996: "a girl wearing blue jeans."

Source image 2341744. First relevant image: CLIP rank 99 -> CLIP+Graph rank 3 (change +96).

**first relevant image: image 2341744** (CLIP rank 99, CLIP+Graph rank 3; clip 0.1848, graph 0.1428, final 0.1428; 16 edges)

Pushed the image up:
- sociable.n.01 -[in]-> image 2341744 (+0.0023)
- main.n.01 -[in]-> image 2341744 (+0.0022)
- course.n.01 -[in]-> image 2341744 (+0.0016)
Pulled the image down:
- adolescent.n.01 -[in]-> image 2341744 (-0.0029)
- student.n.01 -[in]-> image 2341744 (-0.0018)
- friend.n.01 -[in]-> image 2341744 (-0.0014)
Deletion check (remove 3 supporting edges): graph score 0.1428 -> 0.1357, CLIP+Graph rank 3 -> 3.
Without any edge (16 removed): graph score 0.1428 -> 0.1369, CLIP+Graph rank 3 -> 3.

### Query 14193: "the grass beside the car"

Source image 2368850. First relevant image: CLIP rank 98 -> CLIP+Graph rank 3 (change +95).

**first relevant image: image 2368850** (CLIP rank 98, CLIP+Graph rank 3; clip 0.2001, graph 0.1761, final 0.1761; 16 edges)

Pushed the image up:
- sedan.n.01 -[in]-> image 2368850 (+0.0014)
- caption.n.01 -[in]-> image 2368850 (+0.0012)
- ready.n.01 -[in]-> image 2368850 (+0.0009)
Pulled the image down:
- pursuit.n.01 -[in]-> image 2368850 (-0.0022)
- police.n.01 -[in]-> image 2368850 (-0.0017)
- policeman.n.01 -[in]-> image 2368850 (-0.0017)
Deletion check (remove 3 supporting edges): graph score 0.1761 -> 0.1722, CLIP+Graph rank 3 -> 3.
Without any edge (16 removed): graph score 0.1761 -> 0.1603, CLIP+Graph rank 3 -> 8.


## Failure queries

### Query 628: "scattered waves in water"

Source image 2798. First relevant image: CLIP rank 7 -> CLIP+Graph rank 100 (change -93).

**first relevant image: image 2798** (CLIP rank 7, CLIP+Graph rank 100; clip 0.2372, graph 0.0930, final 0.0930; 16 edges)

Pushed the image up:
- current.n.01 -[in]-> image 2798 (+0.0023)
- view.n.02 -[in]-> image 2798 (+0.0016)
- screen_saver.n.01 -[in]-> image 2798 (+0.0014)
Pulled the image down:
- blue_sky.n.01 -[in]-> image 2798 (-0.0018)
- marsh.n.01 -[in]-> image 2798 (-0.0016)
- sky.n.01 -[in]-> image 2798 (-0.0014)
Deletion check (remove 3 supporting edges): graph score 0.0930 -> 0.0868, CLIP+Graph rank 100 -> 100.
Without any edge (16 removed): graph score 0.0930 -> 0.1043, CLIP+Graph rank 100 -> 96.

**rank 1 under CLIP+Graph: image 2353055** (CLIP rank 4, CLIP+Graph rank 1; clip 0.2452, graph 0.2300, final 0.2300; 16 edges)

Pushed the image up:
- competition.n.01 -[in]-> image 2353055 (+0.0023)
- action.n.01 -[in]-> image 2353055 (+0.0021)
- fun.n.01 -[in]-> image 2353055 (+0.0015)
Pulled the image down:
- swell.n.01 -[in]-> image 2353055 (-0.0022)
- surf.n.01 -[in]-> image 2353055 (-0.0021)
- surfboard.n.01 -[in]-> image 2353055 (-0.0013)
Deletion check (remove 3 supporting edges): graph score 0.2300 -> 0.2230, CLIP+Graph rank 1 -> 1.
Without any edge (16 removed): graph score 0.2300 -> 0.2297, CLIP+Graph rank 1 -> 1.

### Query 22323: "tall green and tan grass"

Source image 2405444. First relevant image: CLIP rank 15 -> CLIP+Graph rank 96 (change -81).

**first relevant image: image 2405444** (CLIP rank 15, CLIP+Graph rank 96; clip 0.2174, graph 0.0306, final 0.0306; 16 edges)

Pushed the image up:
- body_waste.n.01 -[in]-> image 2405444 (+0.0015)
- scrape.n.01 -[in]-> image 2405444 (+0.0014)
- plot.n.01 -[in]-> image 2405444 (+0.0013)
Pulled the image down:
- grass.n.01 -[in]-> image 2405444 (-0.0031)
- ditch.n.01 -[in]-> image 2405444 (-0.0014)
- grave.n.01 -[in]-> image 2405444 (-0.0013)
Deletion check (remove 3 supporting edges): graph score 0.0306 -> 0.0257, CLIP+Graph rank 96 -> 96.
Without any edge (16 removed): graph score 0.0306 -> 0.0518, CLIP+Graph rank 96 -> 94.

**rank 1 under CLIP+Graph: image 2410837** (CLIP rank 3, CLIP+Graph rank 1; clip 0.2519, graph 0.2051, final 0.2051; 16 edges)

Pushed the image up:
- course.n.01 -[in]-> image 2410837 (+0.0017)
- gazelle.n.01 -[in]-> image 2410837 (+0.0017)
- antelope.n.01 -[in]-> image 2410837 (+0.0016)
Pulled the image down:
- grassland.n.01 -[in]-> image 2410837 (-0.0032)
- savanna.n.01 -[in]-> image 2410837 (-0.0016)
- graze.n.02 -[in]-> image 2410837 (-0.0016)
Deletion check (remove 3 supporting edges): graph score 0.2051 -> 0.1994, CLIP+Graph rank 1 -> 1.
Without any edge (16 removed): graph score 0.2051 -> 0.2031, CLIP+Graph rank 1 -> 1.

### Query 22494: "wheel of car"

Source image 2406229. First relevant image: CLIP rank 15 -> CLIP+Graph rank 96 (change -81).

**first relevant image: image 2406229** (CLIP rank 15, CLIP+Graph rank 96; clip 0.2075, graph 0.0020, final 0.0020; 16 edges)

Pushed the image up:
- utility.n.01 -[in]-> image 2406229 (+0.0029)
- valve.n.01 -[in]-> image 2406229 (+0.0015)
- drain.n.01 -[in]-> image 2406229 (+0.0009)
Pulled the image down:
- drinking_fountain.n.01 -[in]-> image 2406229 (-0.0025)
- fireplug.n.01 -[in]-> image 2406229 (-0.0020)
- spout.n.01 -[in]-> image 2406229 (-0.0010)
Deletion check (remove 3 supporting edges): graph score 0.0020 -> -0.0041, CLIP+Graph rank 96 -> 97.
Without any edge (16 removed): graph score 0.0020 -> 0.0331, CLIP+Graph rank 96 -> 82.

**rank 1 under CLIP+Graph: image 2363372** (CLIP rank 1, CLIP+Graph rank 1; clip 0.3083, graph 0.2078, final 0.2078; 16 edges)

Pushed the image up:
- sedan.n.01 -[in]-> image 2363372 (+0.0027)
- car.n.01 -[in]-> image 2363372 (+0.0019)
- car.n.02 -[in]-> image 2363372 (+0.0019)
Pulled the image down:
- hubcap.n.01 -[in]-> image 2363372 (-0.0030)
- car_tire.n.01 -[in]-> image 2363372 (-0.0015)
- tire.n.01 -[in]-> image 2363372 (-0.0014)
Deletion check (remove 3 supporting edges): graph score 0.2078 -> 0.2003, CLIP+Graph rank 1 -> 1.
Without any edge (16 removed): graph score 0.2078 -> 0.2068, CLIP+Graph rank 1 -> 1.

### Query 13958: "a wooden sign"

Source image 2367684. First relevant image: CLIP rank 12 -> CLIP+Graph rank 92 (change -80).

**first relevant image: image 2367684** (CLIP rank 12, CLIP+Graph rank 92; clip 0.2623, graph 0.0014, final 0.0014; 16 edges)

Pushed the image up:
- rung.n.01 -[in]-> image 2367684 (+0.0041)
- tongs.n.01 -[in]-> image 2367684 (+0.0023)
- kickstand.n.01 -[in]-> image 2367684 (+0.0019)
Pulled the image down:
- bus_stop.n.01 -[in]-> image 2367684 (-0.0027)
- bollard.n.01 -[in]-> image 2367684 (-0.0023)
- stopping_point.n.01 -[in]-> image 2367684 (-0.0018)
Deletion check (remove 3 supporting edges): graph score 0.0014 -> -0.0081, CLIP+Graph rank 92 -> 95.
Without any edge (16 removed): graph score 0.0014 -> 0.0721, CLIP+Graph rank 92 -> 34.

**rank 1 under CLIP+Graph: image 2363366** (CLIP rank 1, CLIP+Graph rank 1; clip 0.3078, graph 0.1511, final 0.1511; 16 edges)

Pushed the image up:
- adam.n.01 -[in]-> image 2363366 (+0.0040)
- bric-a-brac.n.01 -[in]-> image 2363366 (+0.0037)
- amour_propre.n.01 -[in]-> image 2363366 (+0.0034)
Pulled the image down:
- street_sign.n.01 -[in]-> image 2363366 (-0.0047)
- signpost.n.01 -[in]-> image 2363366 (-0.0038)
- street_name.n.01 -[in]-> image 2363366 (-0.0032)
Deletion check (remove 3 supporting edges): graph score 0.1511 -> 0.1382, CLIP+Graph rank 1 -> 1.
Without any edge (16 removed): graph score 0.1511 -> 0.2044, CLIP+Graph rank 1 -> 1.

### Query 18381: "handicap parking sign on metal post"

Source image 2388012. First relevant image: CLIP rank 15 -> CLIP+Graph rank 95 (change -80).

**first relevant image: image 2388012** (CLIP rank 15, CLIP+Graph rank 95; clip 0.2465, graph 0.0325, final 0.0325; 16 edges)

Pushed the image up:
- passerby.n.01 -[in]-> image 2388012 (+0.0036)
- utility.n.01 -[in]-> image 2388012 (+0.0035)
- post.n.01 -[in]-> image 2388012 (+0.0017)
Pulled the image down:
- parking_meter.n.01 -[in]-> image 2388012 (-0.0035)
- bollard.n.01 -[in]-> image 2388012 (-0.0023)
- drinking_fountain.n.01 -[in]-> image 2388012 (-0.0019)
Deletion check (remove 3 supporting edges): graph score 0.0325 -> 0.0223, CLIP+Graph rank 95 -> 98.
Without any edge (16 removed): graph score 0.0325 -> 0.0724, CLIP+Graph rank 95 -> 75.

**rank 1 under CLIP+Graph: image 2393216** (CLIP rank 3, CLIP+Graph rank 1; clip 0.2760, graph 0.2141, final 0.2141; 16 edges)

Pushed the image up:
- seller.n.01 -[in]-> image 2393216 (+0.0044)
- customer.n.01 -[in]-> image 2393216 (+0.0042)
- campaign.n.04 -[in]-> image 2393216 (+0.0025)
Pulled the image down:
- parking_meter.n.01 -[in]-> image 2393216 (-0.0054)
- parking_space.n.01 -[in]-> image 2393216 (-0.0040)
- parking_lot.n.01 -[in]-> image 2393216 (-0.0029)
Deletion check (remove 3 supporting edges): graph score 0.2141 -> 0.2012, CLIP+Graph rank 1 -> 1.
Without any edge (16 removed): graph score 0.2141 -> 0.2056, CLIP+Graph rank 1 -> 1.

