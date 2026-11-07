# R19 learned target-plane phase — seed 0 independent acceptance (ggmaster913, Mac CPU), 08.11 ~02:35 MSK

**Run:** Sanya, RTX 4070 Ti, `research/sanya-w51-r19@0f8a21f`.
- model.py and train_r19.py are byte-identical to a07b6e9 after LF normalisation; the diff is only the no-TEST loader `run_r19_notest.py`, the same for both arms.
- Asset 722232793, ZIP sha256 7001d3f7…85db: 391/391 files pass SHA256SUMS.
- ckpt control a35766e0…, tphase 05add78f…

**Independent checks** (scripts in `tphase_head/audit/`):
- CPU replay of both GPU checkpoints on the exported VAL16 targets (`replay_r19.py`):
  - optimizer.step / backward patched to raise, so the forward is one pass with no optimiser;
  - max |Δφ| vs the exported DOE.png is .023 / .029 rad, ≤ 1 uint8 code everywhere (0 codes off by > 1).
- TEST: not decoded. The wrapper reads train + val only, and VAL16 is the only evaluation split. No per-input optimisation.
- Mechanism used:
  - mean |θ| = .57 rad, but θ is **smooth**: mean |∇θ| = .048 rad/px. The teacher target-plane phase roughness is ≈ 3.1.
  - |g_k| = .016 / .022 / .013 / .006.

**Paired VAL16** (`analyze_r12.py` + `audit/seedmean_r19.py`, bootstrap 2000 over quartets):
| | control | tphase | Δ [95% CI] | better |
|---|---|---|---|---|
| P | .6148 | .6193 | **+.0045 [+.0013, +.0078]** | 14/16 |
| SSIM | .5592 | .5629 | +.0038 [−.0053, +.0118] | 14/16 |
| B P | .4531 | .4618 | **+.0087 [+.0054, +.0128]** | **8/8** |
| A P | .7766 | .7769 | +.0003 [−.0036, +.0040] | 6/8 |
| B own | 23/32 | 25/32 | +.25 / quartet [0, +.63] | |
| B margin | .094 | .096 | +.0017 [−.0063, +.0080] | |
| worst-plane P / S | .593 / .508 | .602 / .513 | +.009 / +.005 | |
| B inter-z full / low / band | .694 / .940 / .607 | .675 / .939 / .580 | −.019 / −.001 / −.027 | |
| mask HF / rough | .149 / .486 | .167 / .511 | | |
| bright (PNG > .5) / hot / top | .0684 / .0004 / .0012 | .0544 / .0003 / .0013 | bright −20% (within ±25%) | |

**Visual** (`results/r19_s0_sheet_*.png`):
- B64:016 own goes 2/4 → 3/4 (z4 fixed);
- B64:012 and A64:060 are almost unchanged;
- ghosting is still clearly visible.

**Verdict seed 0: INCONCLUSIVE(+), B-specific.**
- The gain is concentrated where expected: on B (8/8 quartets), with lower B band inter-z, and A is unchanged. That is the signature of the hypothesis, unlike R18's uniform A+B +.004.
- Below the pre-registered gate (ΔP ≥ +.010 and ΔB ≥ +.010). Not promoted.
- Caveat: bright-pixel fraction −20% (mild dimming of highlights, within the pre-set ±25%).
- Seed 1 is pre-registered and dispatched (#51 6714389928).
- Replication rule: S1 ΔP ≥ +.003 and ΔB ≥ +.005, B own ≥ control. Promotion is still only at a 2-seed mean ≥ +.010 / +.010 (not lowered).

**Interpretation:** the network uses the target-plane phase but learns only a smooth, lens/tilt-like θ, not the rough teacher-like θ. This is the same smooth-basin limitation, now in the target plane. That is why the effect is real-looking but small.
