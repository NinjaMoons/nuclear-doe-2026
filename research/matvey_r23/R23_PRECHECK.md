# R23-FOURIER-PHASOR-PRECHECK + NIH-DIAG-CAPACITY-015 — precheck report

**Status: PRECHECK DONE · SCIENCE_GATE=PENDING_GRI_REVIEW.** My recommendation is **NO_GO for a full run as currently specified**: no HF capacity signal was found (details below). Run 2026-11-08 ~03:52 MSK, host `DESKTOP-0GHPCI9`, RTX 5060 8 GB, torch 2.9.0+cu128.

| | |
|---|---|
| Owner | @playerr456 |
| Source | `research/matvey-r23-fourier-phasor@5388c3c` (clean tree), base `research/sanya-native-b-sampler@5e15bf1` |
| Script | `research/matvey_r23/r23_fourier_phasor.py`, sha256 `781dc045…40df` |
| Data | native TRAIN512, manifest `76d2474d…6f1c78`, corpus asset `corpus-4b-win-49b9604.tar.gz` sha `6c6f136b…5701` |
| Decoded splits | `train` only. VAL and TEST were **not** decoded. |
| Budget | ≤100 updates per run, 4 runs |
| Raw output | `precheck_report_s0.json` (full per-step history) |

## Code-level diff vs R8-SCRATCH (`research/native_b_sampler/phasor.py::PhasorStemUNet`, scratch init)
| | R8-SCRATCH | R23 |
|---|---|---|
| phase | `wrap(head(x) + atan2(im, re))` | `atan2(im, re)`: the scalar head is **removed** from the output |
| (re, im) local | `Conv1x1([x16 ; bp8])`, N(0,.5), bias 0 | same module, same seed, same init |
| (re, im) global | none | `SpectralConv(x16 → 2, 32×32 modes ±ky)`: learnable complex Fourier mixing of decoder features |
| (re, im) carriers | none | 16 learnable carrier vectors k ∈ [−π, π] rad/px (up to Nyquist), each with per-pixel target-conditioned complex amplitude from `Conv1x1(x16 → 32)` |
| params | 452,531 | 584,179 |

Shared settings:
- Backbone (STEM 12ch, W16, depth 3), seed 0, R3 objective, Adam lr 1e-3, batch 4 and the sampler `randperm(512, gen(seed))[:4]` are identical.
- One forward pass, phase-only, no optimiser at inference.
- The oscillations are learned (k and amplitudes are trainable and target-conditioned), not fixed.

## Results (seed 0, TRAIN only)
### 1. Fixed TRAIN-B overfit, 100 updates, B64:000/001/004/006, same optical forward
Columns are physical Pearson on these 4 quartets, the HF energy fraction (exp(iφ) spectrum beyond 0.25 cyc/px) and |∇φ| in rad/px.

| step | R8-SCRATCH P / HF / ∇φ | R23 P / HF / ∇φ |
|---|---|---|
| 1 | .170 / .062 / .30 | .127 / .207 / .60 |
| 25 | .399 / .085 / .35 | .404 / .095 / .39 |
| 50 | .433 / .131 / .44 | .441 / .140 / .47 |
| 75 | .454 / .156 / .48 | .463 / .164 / .51 |
| 100 | .467 / .185 / .53 | **.479** / .184 / .55 |

Final values (eval mode):
- P: .4666 vs **.4776** (Δ +.011);
- R3 loss: .6162 vs .6037;
- per-plane P: R23 is higher on 3 of 4 planes;
- cross-sample HF coherence: .021 vs .017. HF content is target-specific, not a shared texture, in **both** arms.

### 2. Sampler smoke on TRAIN512, 100 updates
- R3 loss .394 vs .382.
- P on the same TRAIN-B4 set: .375 vs .392.
- Grad norm max: 2.48 vs 3.05, all finite.

### 3. Resources
- Speed: 0.036–0.047 s/update.
- Peak VRAM: 313 MiB (8 GB GPU, batch 4, no reduction needed).
- 12k updates ≈ 8 min training. Predicted p90 with official VAL16 eval: **≤ 15 min** (card limit 35).

## Interpretation (NIH-DIAG-CAPACITY-015)
- **Expressivity vs optimisation.**
  - The random carriers inject HF at init (HF .207, |∇φ| .60).
  - The optimiser **removes** it within 25 steps (.095).
  - HF then grows back on exactly the same trajectory as R8-SCRATCH, ending at .184 vs .185.
  - So the extra spectral/carrier capacity is **not** used to make more HF. Within 100 steps the HF level is set by optimisation from the smooth start, not by head expressivity.
  - Both arms stay far below the teacher's coherent ~2–3 rad/px.
- **Small P gain.** The +.011 P on 4 fixed TRAIN-B quartets (+.017 on the sampler smoke) is one seed and 100 steps of TRAIN overfit. It is **not** evidence of VAL generalisation. Its size is comparable to the init-dependent spread seen at step 1–25.
- **Gate criteria.** The card requires positive train-only evidence of *coherent controllable HF*. That is **absent**: HF and ∇φ match the control. Under the card's rule ("capacity signal absent → NO_GO") my recommendation is **NO_GO for the 12k run in this form**.
  - A variant would need an explicit HF-promoting mechanism, e.g. carrier init kept from being washed out or a teacher-HF-matched objective. That is a different factor and needs its own review.
- **Tests.** No VAL/TEST was decoded, no full training was run, and protected V1/V2/4ch and main were not touched.

## Reproduce
```powershell
$env:PYTHONPATH = "<worktree>"
<venv>\Scripts\python.exe research\matvey_r23\r23_fourier_phasor.py precheck --corpus C:\nuclear-data-49b9604 --out <NEW_DIR> --seed 0 --updates 100
```
