<!-- mvr-41-wc-pilot-long-native-20261107 -->
## @MVRto01 · #41 · WC audit **PASS** · paired pilot **WC_NEUTRAL** (WC dropped) · long native training **LONGER_HELPS** · new research candidate **beats #40 on every VAL quartet**

Host: RTX 4070 Ti, `.venv-41` (py3.11.9, torch 2.9.0+cu128). One process on the GPU at a time.
- **Not done:** TEST opened, the #40 package changed, protected `best.pt` touched, merge, official submission. **There is no submission receipt.**
- Code: branch `research/sanya-whitecollars-stem`, commits `90a8eee` → `1232ef2`.
- Pre-registration was committed **before** each run: `research/wc_stem_pilot/PLAN_WC_PILOT.md`, then `PLAN_LONG_NATIVE.md`.

### REPORT B — WC audit: **WC_DATA_AUDIT_PASS** (rights: see #42, author statement PENDING)
- Archive SHA `f58db14e…c29fc521` matches (readback in [#42](https://github.com/NinjaMoons/nuclear-doe-2026/issues/42#issuecomment-6711454897)).
- 17 588 PNG files, all L8 128×128. The manifest lists 4397 quartets: train 4096, val 200 (100 + 100 stress), test 100, diagnostic 1.
- Clean on every check: **0** forbidden or unsafe files, **0** TEST overlap by SHA, **0** duplicates or z-permutations, **0** exact matches with our TRAIN/VAL.
- dHash raised 6713 near-hits. 6705 were rejected by pixel comparison. The 8 suspects are all against `B208:105#z1` with Pearson 0.19–0.46, so they are not duplicates; they were excluded anyway.
- → **4025 TRAIN quartets** + a 64-quartet holdout.
- Fixes in the audit itself: a nested `stress/manifest.json` had shadowed the root manifest, and the 6 MB JSON was being truncated before parsing. Both fixed; regression test added, 21 tests total.
- The distribution differs a lot from ours: little HF power (0.03 against 0.34 for A and 0.20 for B), more black, more full-range grayscale.

### REPORT C — paired pilot ARM A native512 vs ARM B native512 + WC 50/50
- Setup: STEM 12ch, R3, lr 1e-3, batch 4, fp32, seeds 0/1, 4000 updates.
- Evaluation follows the official path: DOE.png u8 → reload → Fresnel → PNG → reload.
- Numbers are B − A, seed mean, with a quartet bootstrap:

| step 4000 | dPearson [95% CI] | dSSIM [95% CI] |
|---|---|---|
| VAL16 | **−0.0065** [−0.009, −0.003] | −0.0081 [−0.013, −0.004] |
| orgsets10* | **−0.0157** [−0.024, −0.008] | −0.0157 [−0.026, −0.007] |
| WC holdout64 | +0.0252 [+0.020, +0.031] | +0.0167 [+0.012, +0.022] |

- At step 1000, VAL16 is worse by −0.022.
- **Verdict by the pre-registered rule: `WC_NEUTRAL`.** The direction is negative on our VAL and on orgsets; WC helps only on its own holdout. **WhiteCollars is not used further.**
- WC-trained weights stay on this PC and are not published anywhere.

\* `orgsets10` = `doe_multiplane_test_sets.zip` (SHA `ac908518…856c`), 10 hand-made 128×128 L8 target quartets: symbols 7/8/F/K, geometry, gratings, spots, F-controls, grayscale, contours. Alexander provided it locally. **Evaluation only**: no training on it and no checkpoint selection. 0 SHA overlap with our manifest.

### Positive result: longer native training (pre-registered addendum, only factor = number of updates)
Native-only training, 12000 updates (about 4 minutes on the GPU); step 12000 against step 4000, seed mean:
- VAL16: P +0.0097, SSIM +0.016.
- orgsets10: P +0.012.

→ **`LONGER_HELPS`.**

| VAL16 / orgsets10, official path | P | SSIM | P orgsets | SSIM orgsets | A P | B P | worst-plane P | B own-target |
|---|---|---|---|---|---|---|---|---|
| #40 `569e1700` (s0 step900) | 0.5778 | 0.5067 | 0.4562 | 0.2642 | 0.7298 | 0.4257 | 0.5495 | 24/32 |
| **new s0 step12000 `19b05dad`** | **0.6124** | **0.5536** | **0.5100** | **0.3217** | 0.7733 | 0.4516 | 0.5917 | 25/32 |
| new s1 step12000 `240663f6` | 0.6045 | 0.5457 | 0.4998 | 0.3143 | 0.7625 | 0.4465 | 0.5883 | 25/32 |

**Paired comparison of s0 step12000 against #40, per quartet:**
- VAL16: dP **+0.035** [+0.028, +0.042], dSSIM **+0.047** [+0.032, +0.066], **16/16 quartets better**.
- orgsets10: dP +0.054 [+0.032, +0.079], **10/10 better**.
- WC holdout: dP +0.046, **64/64 better**.

Seed 1 also beats #40 on every VAL and orgsets quartet.

**Checks of the new checkpoint:**
- Through the **official CPU one-forward CLI** (`infer_neural_stem`) on VAL16: P 0.61243 / SSIM 0.55356, the same as the GPU evaluation.
- Inside the **#40 package runtime** as a drop-in replacement:
  - `test_stem` OK; `test_infer` + `test_package` 9 tests OK;
  - `acceptance` rc=0, one forward, 2.0 s per process on CPU;
  - A64:006: P 0.83 / 0.85 / 0.86 / 0.85.
  - The package itself is not modified (SHA256SUMS 96/96).
- Architecture and objective are the same as #40: 12ch STEM + R3, no new channels and no Adam at inference. Only the training length differs.

**Honest caveats (#47):**
- *Ghosting is not fixed.* On orgsets and VAL B, every model, #40 included, shows an overlay of all 4 targets in each plane; the 4 distinct pictures are not separated. B inter-z of the reconstructions is 0.69, against 0.05 in the targets.
- The gain from length is in sharpness and contrast of the right picture, not in separating the planes.
- GPU training is not bit-deterministic: repeating s0 step4000 gave 0.6016 against 0.6004.
- Choosing s0 over s1 is a choice by VAL16; s1 is also better than #40 on every quartet.
- The contact sheets are in the asset: `analysis/long_sheet_*.png`, raw PNG with no stretching.

### REPORT D — handoff
- **Asset `721613794` `desktop41-native-long-mvr.zip`**, 18 072 257 B, SHA256 `5289fe16778b909ccb50575b23c84c8d54a6a169fdd72205cc11f01a171feac3` (server digest matches). 39 files plus `SHA256SUMS.txt`:
  - 4 native checkpoints (s0/s1 × 4000/12000);
  - configs and train logs of all 6 runs;
  - eval JSON (per-quartet);
  - analysis, separation and contact sheets;
  - CPU replay and #40-runtime acceptance receipts;
  - WC audit summary;
  - pre-registration plans.
  - **No WC images and no WC weights.**
- **Proposal to @NinjaMoons / Pavel (decision is yours; I am not building or submitting anything):**
  - Replacing `model/stem_seed0_step_000900.pt` in the #40 package with `stem_native_s0_step_012000.pt` (`19b05dad…`) needs no code changes and gives +0.035 P / +0.047 SSIM on VAL.
  - If approved, @playerr456 can re-run his #40 acceptance with this file. I am ready to check it independently on this PC.
- **#43** (A/B 25/75): the GPU is free. Per #43/6710870678 I can start it now on a 12000-update base, if that is still relevant after this result.
