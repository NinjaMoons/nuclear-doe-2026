# NIH-DIAG-C3-OVERFIT — C3/C4 · **VERDICT=AMBIGUITY** · C4 coherent HF = **YES**

| | |
|---|---|
| Host | `DESKTOP-0GHPCI9` RTX 5060 |
| Start | 04:27:25 MSK; 179 s total |
| Source | `research/matvey-r23-fourier-phasor@a4222d7` (`diag_c3_c4.py`, sha256 `9a794cc9…6b26`) |
| Data | native TRAIN512 `76d2474d…`. Only `train` was decoded; no VAL/TEST/WC. |
| Arm | R8-SCRATCH (fresh STEM 12ch W16, phasor head N(0,.5)), R3, Adam lr 1e-3, seed 0 |
| N64 set | first 64 TRAIN-B in manifest order (B64:000 … B208:015) |
| Raw output | `diag_c3_c4.json` |

Metrics are on B64:000 after **DOE u8 quantisation** (φ = v/255·2π−π) → propagate → 4 amplitude planes /max → **u8 rounding**:
- Pearson and global SSIM (L=1);
- own 4×4 = rec plane argmax over target planes;
- HF = energy fraction of FFT(e^{iφ}) at r > .5 Nyquist;
- |∇φ| = mean wrapped neighbour difference.

This is an in-memory re-implementation of the official u8 path, not `export_run` PNG files on disk.

## Results on B64:000
| run | P | SSIM | own 4×4 | HF r>.5Nyq | \|∇φ\| rad/px | per-plane P |
|---|---|---|---|---|---|---|
| N1 (1 quartet, 3000 upd) | **.6562** | .5201 | 4/4 | **.590** | 1.24 | .701/.698/.529/.697 |
| N64 (64 TRAIN-B, batch 4, 3000 upd) | .4938 | .4220 | 4/4 | .241 | 0.64 | .423/.572/.399/.581 |
| ORACLE from random φ (Adam .05, 1000) | .6624 | .5206 | 4/4 | .772 | 1.54 | .719/.701/.514/.716 |
| ORACLE from N1 output | **.6704** | .5270 | 4/4 | .668 | 1.38 | .724/.709/.531/.718 |

N64 on B64:001: P .4501 / SSIM .3736 / own 2/4 / HF .201.

## C4: swap HF band (r > .5 Nyq of FFT(e^{iφ})) between the N64 predictions for B64:000 and B64:001
| quartet | P original | P after swap | ΔP |
|---|---|---|---|
| B64:000 | .4938 | .3726 | **−.121** |
| B64:001 | .4501 | .3415 | **−.109** |

The HF band of each prediction is **target-specific and coherent**: ΔP ≤ −.02 on both quartets.

## Pre-registered reading
- **EXPRESSIVITY** (N1 ≤ ORACLE − .05 **and** N1 HF ≤ .3): **false**.
  - N1 reaches .6562 vs the best oracle .6704 (gap .014).
  - N1 HF is .59; |∇φ| is 1.24 rad/px, in the oracle range.
  - The W16 R8-SCRATCH network *can* represent near-oracle coherent HF phase for one target.
- **AMBIGUITY** (N1 ≥ ORACLE − .02 **and** N64 ≤ N1 − .03): **true**.
  - .6562 ≥ .6504, a margin of only .006.
  - N64 is .163 below N1.
  - Its HF collapses to .24 and |∇φ| to .64, i.e. the network averages towards a smooth phase when it must serve 64 targets.
- **C4:** coherent HF = **true**.
- **Caveats:**
  - Single seed and single quartet. The AMBIGUITY condition holds with a .006 margin.
  - N64 got 3000 updates at batch 4, about 188 passes per quartet, vs 3000 passes for N1. Part of the N64 gap may be an update-budget effect, not pure ambiguity. A budget-matched N64 (≈48k updates) or a seed-1 repeat would harden the verdict.
  - Diagnostic only; no candidate, no promotion.
