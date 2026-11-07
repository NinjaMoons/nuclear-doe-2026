# #43 · native-only B-focused sampler fine-tune from frozen s0step12000 — pre-registered

Written 2026-11-07 ~22:35 MSK BEFORE any run. Host DESKTOP-T1K698V RTX 4070 Ti, `.venv-41` (py3.11.9, torch
2.9.0+cu128). Branch `research/sanya-native-b-sampler` (from `5e6300f`). Directive #43/6711867567.

## Fixed
- Init: frozen `stem_native_s0_step_012000.pt` sha `19b05dad34601a8b5ea4c78e7cd6904b0c639a218f8fbcdeabc2afbb5d6f083f`
  (#41/6711603518, cross-host audit PASS #41/6711679422). Same weights in both arms, fresh Adam lr 1e-3, batch 4,
  fp32, R3 objective, official Fresnel, CUDA, cudnn deterministic.
- Data: native TRAIN512 only (manifest `76d2474d…`; 256 A + 256 B by manifest `kind`). No WC, no TEST, no external.
- Arms (only factor = A/B sampling distribution):
  - **C50** control = original driver sampler `randperm(512, gen)[:4]` (A/B 50/50 in expectation);
  - **B75** = per update 1 A `randperm(256 A, gen)[:1]` + 3 B `randperm(256 B, gen)[:3]` (A25/B75, uniform within
    group, no repeats inside a batch).
- Sampler seeds 0 and 1 (gen seed = 20000 + s). Updates 1000, checkpoints at 100, 250, 500, 1000.

## Stages
- Stage 0: assert 512 eligible TRAIN ids (256/256), VAL16 ids locked, init sha; step-0 eval of the frozen model on
  the same path (must reproduce VAL16 P .6124). 100-step health: finite loss/grad, sample counts.
- Stage 1: **primary readout = step 500**, seed mean, B75 − C50. Step 1000 is reported; it may be used only if the
  step-500 verdict is positive and the same direction holds at 1000.

## Decision (fixed now)
- `B75_HELPS` if at step 500: dP_B(VAL16) >= +0.01 AND dP_A >= -0.01 AND dP_total >= 0 AND dSSIM_total >= -0.005
  AND B own-target count not lower. `B75_HURTS` if dP_total <= -0.005 or dP_A < -0.02. Otherwise `B75_NEUTRAL`.
- Promotion over `19b05dad` (research candidate only) needs: B75 helps, the checkpoint beats `19b05dad` on VAL16 P
  and SSIM and on orgsets10 P, no worse worst-plane, raw sheets not visually worse. Otherwise `19b05dad` stays.
- Readouts: VAL16 total/A/B/per-z/worst, own-target 4x4 + inter-z (`separation.py`), orgsets10, WC holdout64 (eval
  only), paired quartet bootstrap. Raw contact sheets on the predeclared quartets **A64:060 (A typical), B64:012
  (B median), B64:016 (B hard)** = the #47 set; rows TARGET / frozen #40 `569e1700` / frozen `19b05dad` / arm.
- Budget: <= 30 min wall total; one GPU process at a time; stop by 02:00 MSK regardless.
