# Independent acceptance R19-TPHASE-S0 (NIH-SCIENCE-R19-REVIEW-004) · ggmaster913 · 08.11.2026 ≈02:35 МСК

**Verdict.** **INCONCLUSIVE(+), NOT_PROMOTED.** V2 (19b05dad…) remains the baseline. R19-TPHASE-S1: **SCIENCE_GATE = NO_GO**.

## Provenance and integrity
- **Asset 722232793** `r19-s0-desktop-mvr.zip`: SHA256 `7001d3f7…85db` matches. `SHA256SUMS.txt` checks 391/391 OK.
- **Code:** `research/sanya-w51-r19@0f8a21f` = `a07b6e9` + `run_r19_notest.py`.
  - The wrapper only replaces the loader with a train+val loader that never reads TEST.
  - `script_sha256` / `model_sha256` in the summary are the CRLF versions of `a07b6e9:tphase_head/{train_r19,model}.py` (checked: e430096a…, 7091223c…).
- **No per-input optimisation at inference.** In `model.py` the only `no_grad` is in init. Optimizer and backward appear only in the training loop of `train_r19.py`. Evaluation is the stock `tm.evaluate_checkpoints`.
- **Independent Mac CPU replay** (`replay_r19.py`, code `a07b6e9`, VAL16 only, TEST not read, per-file SHA of the manifest checked):

| arm | Mac P / SSIM | GPU P / SSIM | PNG |
|---|---|---|---|
| control | .614817 / .559151 | .614819 / .559163 | targets identical; DOE and recon ≤ ±1 code (6 % of recon pixels, CUDA vs CPU rounding) |
| tphase | .619327 / .562930 | .619321 / .562913 | same |

  Checkpoints: control `a35766e0…`, tphase `05add78f…`.

## Paired analysis (official uint8 save/reload, VAL16, `analyze_paired.py`)

| | control | tphase | Δ |
|---|---|---|---|
| P (all) | .6148 | .6193 | **+.0045**, 95 % CI [+.0013, +.0078], 14/16 quartets better |
| SSIM (all) | .5592 | .5629 | +.0038, CI [−.0049, +.0115], 14/16 better |
| B P / SSIM | .4531 / .3682 | .4618 / .3763 | +.0087 / +.0081 |
| B own argmax | 23/32 | 25/32 | +2; per quartet +.25, CI [0, +.5] |
| B margin | .094 | .096 | +.002, CI [−.007, +.008] |
| inter-z B full / band | .694 / .607 | .675 / .580 | −.019 / −.027 |
| worst plane P / SSIM | .593 / .508 | .602 / .513 | +.009 / +.005 |
| mask HF (r > .5) / roughness | .149 / .486 | .167 / .511 | mechanism is used: moves towards the teacher, but far from HF .6–.7 |
| lit (>0.5) / hot / top | .068 / .0004 / .0012 | .054 / .0003 / .0013 | lit −20 % (within ±25 %) |

**Sheets** `r19_sheet_B64_012.png`, `r19_sheet_B64_016.png`:
- visually almost identical to control;
- B64:016 z4 becomes own (0.43→0.44 against foreign 0.44→0.43);
- ghosting (rings plus the diagonal) persists in both arms.

## Gate (pre-registered in PLAN_R19)
- **SIGNAL requires ΔP ≥ .010 and ΔB ≥ .010.** Not met: ΔP = +.0045, ΔB = +.0087.
- **All safety conditions are met:** own, worst plane, SSIM, inter-z, lit/hot/top.
- **Outcome:** INCONCLUSIVE(+).

**Noise floor.** The matched controls of one recipe give P .6112 (R18 s0, Mac CPU), .6130 (R18 s1) and .6148 (R19 s0, GPU). The run-to-run spread ≈ .002–.004 is comparable to the effect. The paired quartet CI does not account for training-run variance. Precedent: R18 s0 gave +.0041 and s1 gave +.0004.

## Seed 1 decision: NO_GO
1. **The pre-registration** runs seed 1 only on SIGNAL.
2. **No outcome of s1 changes the decision.** Reaching mean ΔP ≥ .010 would need s1 ≥ +.0155, about 3.5× the s0 effect.
3. **One extra seed cannot separate +.004 from the noise floor** (see R18). That would take ≥ 4–5 seeds, which does not fit before 04:30, and the gate does not lower the +.010 threshold.

Science claim for the defense: "learned target-plane phase gives a small, mechanistically consistent effect (+.0045 P, +2 own on B, inter-z −.02). It is not confirmed by a second seed and not promoted."
