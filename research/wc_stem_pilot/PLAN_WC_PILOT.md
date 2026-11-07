# #41 paired GPU pilot — STEM 12ch + R3: native512 vs native512 + WhiteCollars TRAIN (pre-registered)

Written BEFORE any training run (2026-11-07 ~21:45 MSK), host DESKTOP-T1K698V RTX 4070 Ti, venv `.venv-41`
(py3.11.9, torch 2.9.0+cu128), code = this branch `research/sanya-whitecollars-stem` (base `44f0378`).

## Data
- Native: `manifest_expanded.json` file sha `76d2474d…6f1c78`; TRAIN 512 (256 A + 256 B) in manifest order, VAL16
  (8 A + 8 B). TEST is never decoded (my corpus copy has no TEST images at all).
- WhiteCollars: archive sha `f58db14e…c29fc521`; audit `WC_AUDIT.json` technical PASS, 0 blocking, 0 TEST overlap;
  only `split=="train"` quartets of `data_gray_4096/manifest.json`, minus flagged and minus the fixed 64-quartet
  internal holdout -> **4025 TRAIN quartets** (adapter `WhiteCollarsTrain`, u8/255 amplitude, z order 01..04 =
  24.1/25.3/26.5/27.7 mm = our z1..z4). WC validation/test/stress/diagnostic: never used.
- Rights: training basis = explicit instruction of Pavel (#42/6710871354, #43/6710870678) and of Alexander (local
  operator); written author statement still PENDING -> results are RESEARCH ONLY, nothing goes into the contest
  package without coordinator approval + rights confirmation.
- Eval-only sets (never trained on, never used for checkpoint selection):
  - `wc_holdout64` (WC internal holdout, in-domain for WC);
  - `orgsets10` = `doe_multiplane_test_sets.zip` (sha `ac908518…856c`), 10 hand-made target quartets
    (symbols, geometry, gratings, spots, F-controls, grayscale, contours); 0 file-sha overlap with our manifest.

## Arms (only factor = training data mix)
Both arms: `build_stem_model(seed)` (12ch back-projection STEM), R3 objective (`train_wrapper.research_objective`),
Adam lr 1e-3, batch 4, fp32, cudnn deterministic, official Fresnel/forward, CUDA.
- **A native**: per update `randperm(512, gen(seed))[:4]` — identical sampler to the stock driver.
- **B native+WC 50/50**: per update `randperm(512, gen(seed))[:2]` (= the first 2 native indices of arm A at the same
  step) + `randperm(4025, gen(10000+seed))[:2]` WC quartets.
- Seeds 0 and 1. **4000 updates** each, checkpoints every 500. No LR/loss changes, no early stop.

## Readouts
- Official path per quartet: phase -> DOE.png uint8 -> reload -> 4x Fresnel -> 4 reconstructed PNG -> reload ->
  Pearson/SSIM (`train_multiplane.evaluate_checkpoint`).
- Primary: **VAL16 at step 4000**, mean over seeds 0/1. Also reported: step 1000 (= #5 budget), every checkpoint,
  A/B x z, worst plane, own-target argmax / inter-z (`separation.py`), WC holdout64, orgsets10, #40 candidate
  `569e1700` (s0 step900) as a fixed reference, paired quartet bootstrap (10 000 resamples, quartet = unit).
- Raw contact sheets (#47): TARGET / A / B / #40 for VAL A, VAL B, orgsets10. No contrast stretch.

## Decision (fixed now)
- `WC_HELPS` if at step 4000 (seed mean): dP_VAL >= +0.01 AND dSSIM_VAL >= -0.005 AND neither group A nor B has
  dP < -0.02 AND worst-plane dP >= -0.01. `WC_HURTS` if dP <= -0.01. Otherwise `WC_NEUTRAL`.
- A checkpoint becomes a *proposal* (never an automatic replacement of #40) only if it beats `569e1700` on VAL16
  P and SSIM, does not lose on orgsets10 P, and passes the visual sheet. Adoption = Pavel's decision.
- Budget cap: whole pilot <= 60 min wall on this GPU, one process at a time.
