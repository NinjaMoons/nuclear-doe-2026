# R14 report — D4_PHYSICS_PASS (c64) / D4_AUG_NO_GO at the 500-step paired pilot (seed 0), 2026-11-07
Stage A: exact D4 symmetry of the official single-FFT Fresnel / inverse / STEM features / codec for reflections about
the centred index 64 (max rel 3.8e-16); pixel-centre flips are not exact (rel up to 0.50). See D4_PHYSICS_AUDIT.md.
Stage B (PLAN_D4_STAGE_B.md): STEM vs STEM + synchronous c64-D4 training augmentation (loss-equivalent, tested),
seed 0, 500 updates, everything else identical. Arm `none` reproduces R12 arm A and Matvey's STEM at step 500 exactly
(VAL P 0.56913). VAL n=16, step 500 (= rule checkpoint):
| arm | P | SSIM | A P/SSIM | B P/SSIM | B own | B margin | B inter-z full/low/band | worst P/SSIM | HF | hot | top |
| none | .5691 | .4949 | .7220/.6683 | .4163/.3216 | 24/32 | .090 | .681/.941/.621 | .547/.446 | .106 | .0005 | .0009 |
| d4   | .5671 | .4925 | .7194/.6639 | .4148/.3211 | 24/32 | .089 | .655/.938/.596 | .544/.448 | .125 | .0004 | .0009 |
Paired (bootstrap 2000 over quartets): dP -0.0020 [-0.0040,+0.0003] (2/16 better), dSSIM -0.0024 [-0.0092,+0.0036]
(7/16), dB_margin -0.0019 [-0.0053,+0.0029], dB_own 0. Pass criteria 1 and 2 fail -> D4_AUG_NO_GO; no 1000-step/seed-1
extension. Lower B inter-z of d4 is not counted (no gain in own-target identity or quality). Visual (fixed A64:006,
B64:002, negative case B64:016 = lowest own-margin in `none`): d4 and none are visually equivalent; on B64:016 both
show the z3 line and z2/z4 rings on every plane and lose the z1 gradient.
Limitations: one seed, 500 updates, one augmentation policy (uniform over 8 elements every step).
