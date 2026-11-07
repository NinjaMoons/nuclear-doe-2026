# R14 Stage B — pre-registered (before any Stage-B run), 2026-11-07 ~21:35 MSK
One factor: arm `d4` = STEM + per-quartet synchronous exact-D4 (c64) augmentation during training (phi = g^-1 STEM(g T),
loss-equivalent to training on g(T), tested), arm `none` = STEM. Everything else identical: driver tm.main via
d4_stem.train_wrapper_d4, manifest 76d2474d (native 512 TRAIN, original sampling), R3 objective, lr 1e-3, batch 4, fp32,
CPU 4 threads, seed 0, same init (tested), same batch indices (driver generator), eval-every 100, VAL-only evaluation.
Augmentation generator: private torch.Generator(4545 + seed); global RNG untouched (tested).
Steps: health d4 100 updates -> paired none/d4 500 updates. Extension to 1000 + seed 1 only if pass and time allows.
Checkpoint for comparison: step 500 (final) and rule 6697684744; report both.
Pass (all): (1) B own-margin mean d4 >= none + 0.01 OR B own-argmax d4 > none; (2) mean SSIM d4 >= none;
(3) A-group P and SSIM d4 >= none - 0.01; (4) worst-plane P/SSIM d4 >= none - 0.01; (5) no speckle/brightness degradation
(hot px, top-pixel power <= 1.5x, level >= 0.8x); (6) visual: not worse on fixed examples. Else D4_AUG_NO_GO/INCONCLUSIVE.
Fixed visual examples (#47): VAL A64:006, VAL B64:002 (first A/B keys, same as R12) and the visual negative case =
the B VAL quartet with the lowest own-margin in arm `none` at step 500. Raw 4x3 TARGET/STEM/D4 sheets with z, no
per-image contrast changes, plus 4x4 own-target matrices.
