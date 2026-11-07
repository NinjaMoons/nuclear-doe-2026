# #43 result (B75) + diagnostic oracle — 2026-11-07 ~22:55 MSK

## B75 vs C50 fine-tune from frozen 19b05dad (PLAN_B75.md): **B75_NEUTRAL**
Seed mean (0,1), VAL16 official path. Step 500 (primary): dP -0.0034 [CI -0.0047,-0.0021], dSSIM -0.0059,
dP_A -0.0048, dP_B -0.0020, worst-plane +0.0003; B own-target 25-26/32 in both arms. Step 1000: all |d| <= 0.0023.
Neither arm improves on the init (19b05dad: P .6124). Wall: 20-21 s per 1000-update arm, eval ~14 s/checkpoint.

## Why B still overlaps — diagnostics (VAL16 only, no TEST, oracle is NOT a contest method)
| | P | SSIM | B P | B own | B inter-z (phys) |
|---|---|---|---|---|---|
| STEM 19b05dad, one forward | .6124 | .5536 | .4516 | 25/32 | .687 |
| O1 stock Adam contract, rand init, 2000 | .6779 | .6473 | .5392 | 17/32 | .475 |
| O2 Adam on R3 from STEM phase, 2000 | .7952 | .7415 | .6526 | 31/32 | .496 |
Targets B inter-z = .05. Same codec/Fresnel/metrics (export_run). => one phase-only DOE CAN separate B planes far
better than the network: the gap is model-side, not physics.
Fit check: 19b05dad TRAIN64 (every 8th A/B train quartet) P .6198 / SSIM .5729 vs VAL .6124 -> **underfitting**,
not overfitting (consistent with LONGER_HELPS).

## Next (pre-registered here, before running): capacity
Ideas excluded by #43: R13 carrier, R14 D4, R15 axial, HF diffuser, WC. Cost-ranked candidates:
1. width/depth of the same 12ch STEM U-Net (no new code, loader reads hparams, GPU step time ~flat 17-25 ms,
   CPU one-forward 0.02-0.05 s) — **run now**;
2. teacher distillation from per-quartet Adam-R3 phases on TRAIN (new loss, phase-branch alignment) — only if 1 fails.
Arms: W32d3, W64d3, W32d4 (base_channels/depth), native TRAIN512, pilot sampler `randperm(512,gen(seed))[:4]`, R3,
lr 1e-3, batch 4, fp32, seeds 0/1, 12000 updates, checkpoints every 2000. Control = existing W16d3 runs
L_native_s0/s1 (same code path, 12000 updates).
Primary: step 12000 seed mean. `CAPACITY_HELPS` for an arm if dP_VAL16 >= +0.01, dSSIM >= 0, dP_B >= 0,
worst-plane >= control, AND it is also better on BOTH independent eval sets (orgsets10, WC holdout64). Among
passing arms the one with best VAL16 P is the research candidate; then CPU CLI + #40-runtime drop-in check.
