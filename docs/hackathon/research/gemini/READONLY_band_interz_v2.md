# Read-only evidence for G1 (physics vs underfitting) — @ggmaster913, 07.11 ~23:15 MSK

Model: v2 STEM native s0 step12000 `19b05dad…` (package asset 721714285), my Mac CPU replay of VAL16 through the locked
runtime (#41/6711679422). Official recon PNG (A/max uint8), z2..z4 bilinearly resampled to the z1 physical grid
(x = (n-64)*pitch_z, align_corners=True), radial FFT band masks in units of Nyquist; mean pairwise (6 pairs) Pearson
between planes, averaged over 8 quartets per group. Targets resampled the same way.

| group | images | r<=.15 | .15-.3 | .3-.5 | >.5 |
|---|---|---|---|---|---|
| A | targets | +.94 | +.78 | +.50 | +.15 |
| A | v2 recon | +.95 | +.85 | +.36 | -.07 |
| B | targets | +.08 | +.01 | +.00 | +.00 |
| B | v2 recon | **+.94** | **+.77** | +.31 | -.05 |

Reading (CONFIRMED for this model/VAL16): B ghosting is concentrated in low and low-mid spatial frequencies; above
~0.3 Nyquist the planes are nearly independent, as the targets are. Comparison (MODEL_APPROXIMATION — different data,
2 TRAIN-B quartets, coarser bands): the per-sample Adam teacher (#39 Stage0) reached low-pass ~.88-.93 but band
.15-.5 ~.15-.31, i.e. the low band is hard even for direct phase optimisation (physics-limited at this geometry),
whereas the .15-.3 band (v2 .77) is where a learnable gap remains (underfitting). Not a proof of a physical bound.
Script: inline in this commit message context; data: band_interz_v2_val16.json.
