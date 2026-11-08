# R23-FULL-OWNER-S0 — result: **NO_GO (null)** · 2026-11-08 ~04:20 MSK

The run was launched on the host owner's direct instruction **without SCIENCE_GATE=GO**; the precheck had recommended NO_GO. It counts as research evidence only.

| | |
|---|---|
| Host | `DESKTOP-0GHPCI9` (RTX 5060 8 GB) |
| Source | `research/matvey-r23-fourier-phasor@583a31a` (clean) |
| Data | native TRAIN512 (manifest 76d2474d) for training; VAL16 for the official eval |
| Not used | TEST, WhiteCollars |
| Arms | R8-SCRATCH control vs R23, seed 0, 12000 updates, const lr 1e-3, batch 4, R3, sampler `randperm(512, gen(0))[:4]` |
| Wall time | 315 s + 343 s training, ~11 min end-to-end |
| Eval path | official `evaluate_checkpoint` (DOE u8 save/reload → 4 amplitude PNG save/reload → Pearson + global SSIM) |
| Raw output | `eval_val16_s0.json` |

## VAL16 (official)
| step | arm | P | SSIM | A P / SSIM | B P / SSIM |
|---|---|---|---|---|---|
| 4000 | R8-SCRATCH | .6102 | .5586 | .7692 / .7514 | .4513 / .3658 |
| 4000 | R23 | .6094 | .5513 | .7678 / .7371 | .4511 / .3654 |
| 8000 | R8-SCRATCH | .6129 | .5590 | .7737 / .7536 | .4522 / .3645 |
| 8000 | R23 | .6130 | .5564 | .7737 / .7475 | .4524 / .3653 |
| 12000 | R8-SCRATCH | .6160 | .5588 | .7775 / .7538 | .4545 / .3639 |
| 12000 | R23 | **.6168** | **.5628** | .7788 / .7546 | .4547 / .3709 |

## Paired R23 − control, per quartet (bootstrap 95 % CI)
| step | dP | CI | dP wins | dSSIM | dSSIM wins |
|---|---|---|---|---|---|
| 4000 | −.0008 | [−.0027, +.0009] | 7/16 | −.0073 | 6/16 |
| 8000 | +.0001 | [−.0010, +.0012] | 9/16 | −.0026 | 6/16 |
| 12000 (primary) | +.0007 | [−.0015, +.0030] | 8/16 | +.0039 | 12/16 |

- Worst-quartet P at 12k: .4085 vs .4121.
- vs V2 (.61243 / .55356 / B .4516): R23 12k s0 gives +.0043 P, +.0092 SSIM, +.0031 B. This is about the same as the control, so the gain comes from the from-scratch phasor recipe (already known from R8-SCRATCH), not from R23.

## Verdict
- **Gate** (+.010 overall and +.010 on B vs the matched control): **FAIL**. dP +.0007 with a CI spanning 0; dB P +.0002.
- **SSIM** shows a small B-SSIM plus at 12k (+.007), but the sign flips at 4k/8k.
- **Precheck confirmed.** The Fourier/carrier capacity is not converted into coherent HF or into better metrics. This matches the H3 root cause: the HF needs a different training signal, not more head expressivity.
- **No promotion, no seed1.** Checkpoints are kept locally only (sha256 in `eval_val16_s0.json`).

Checkpoint SHA prefixes:

| arm | 4000 | 8000 | 12000 |
|---|---|---|---|
| R23 | `530d9e61` | `0542e6a5` | `16989852` |
| R8-SCRATCH | `3e6187ce` | `cacc69af` | `2087dee2` |
