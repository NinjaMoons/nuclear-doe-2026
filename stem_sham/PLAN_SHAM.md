# P1 STEM sham test (#44 section 4) — pre-registered before any run, 2026-11-07 ~20:35 MSK. DIAGNOSTIC ONLY.

Question: does STEM gain come from target-specific back-projection content (H1) or from per-z positional/chirp geometry
channels (H2)? 100 updates is a diagnostic, never proof.
Arms (12 input channels each; T1..T4 = 2T-1 identical; only the 8 added channels differ; same FourInputUNet body with a
12-ch first block; same seed 0 init for all layers after the stem; same batches/optimizer/R3 objective/lr 1e-3/batch 4):
  A proper: Re/Im of inverse_fresnel(T_k + 0j, z_k) / rms   (exactly back_projection_stem.stem.backprojection_features)
  B chirp:  fixed, target-independent cos/sin of pi (x^2+y^2)/(lambda z_k) on the DOE grid (centred, x=(n-64)*10 um)
  C shuffled: the SAME proper back-projection features, but computed from a DONOR TRAIN quartet: a fixed bank of 32
     TRAIN quartets (first 32 TRAIN keys in manifest order, persistent buffer); donor index = (sum of uint8 codes of the
     input quartet) mod 32; if the donor's targets equal the input (self), use index+1. Correct statistics, wrong mapping.
Run: hf_basis_head-style wrapper, 100 updates, eval VAL at step 100 (official export). Metrics: VAL P/SSIM (ALL/A/B),
B own-argmax, B margin, B inter-z (z1 physical grid). Readout (descriptive): H1 plausible if A > C and A > B on P and
B own-argmax; H2 plausible if B ~ A (|dP| < 0.01) and C ~ A. No full runs, no TEST, no production change.
