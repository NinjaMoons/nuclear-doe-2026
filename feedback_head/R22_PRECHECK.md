# R22_PRECHECK — learned optical-feedback stage 1 on V2 STEM (#51/6714825411) · @MVRto01 · 2026-11-08 03:10 MSK
Status: implementation + static tests + 100-step smoke DONE; **full run waits for @ggmaster913 SCIENCE_GATE=GO**.

## Source
Branch `research/sanya-r22-feedback` (from `research/sanya-w51-r19@0f8a21f` = R19 a07b6e9 + no-TEST host wrapper):
`feedback_head/model.py`, `train_r22.py` (R19 template: same sampler randperm(512, gen(30000+seed))[:4], R3
`bw.research_objective`, Adam lr 1e-3, batch 4, fp32, official `evaluate_checkpoints` VAL16), `run_r22_notest.py`
(data = train+val in manifest order with per-file sha256; TEST never decoded), `test_r22.py`.

## One factor
feedback arm = V2 STEM (stage 0, cloned weights 19b05dad, trainable) + stage-1 U-Net (base 16, depth 3) on 22
deterministic features computed ONCE from phi0: sin/cos phi0, T(4), Ahat_k=|U_k|/max (4), r_k=T_k-Ahat_k (4),
Re/Im of E_k = P_k^-1(r_k * U_k/|U_k|)/rms (8); phi = wrap(phi0 + pi*tanh(head)), head zero-init.
control arm = V2 STEM fine-tune, byte-identical to the R19 control path. Same seed/sampler/steps.
Not an unrolled optimiser: no backward/optimizer/autograd.grad/iteration in forward (test), the correction is
produced by trained weights, not phi0 + alpha*gradient. It does use 4 forward Fresnel + 4 inverse Fresnel inside the
forward pass -> legality vs organizer wording ("multi-stage NN with physics" allowed; per-input NN->Adam rejected) is
for the science lead / coordinator to judge; I do not certify it.

## Tests / smoke (RTX 4070 Ti)
- unittest feedback_head.test_r22: 3/3 OK (init == V2 < 1e-5 rad; finite non-zero grad on corr.head; features
  finite [1,22,128,128]; checkpoint round-trip exact; no optimiser in forward).
- 100 updates each arm (no eval): finite; step-1 objective identical in both arms (.31467 = same start);
  corr.head |w| .059 after 100 updates (correction is used); peak VRAM control 150 MiB, feedback 305 MiB.
- Wall incl. data load: control 6.6 s / 100 upd, feedback 8.8 s / 100 upd -> 6000 upd ≈ 2 + 4.5 min + eval ~1 min
  per arm; both arms p50 ~9 min, p90 ~15 min.

## Run on GO (prepared, not started)
`bash C:/ndoe-out/run_r22_s0.sh` -> C:\ndoe-out\r22_s0_control, r22_s0_feedback (skips an arm whose dir exists).
Gate (from the card, relative to the matched control): ΔP >= +.010 AND ΔB >= +.010, SSIM >= control, worst >= control
-.005, B own >= control, lit/hot/top drift <= 25%. Report: r22_summary.json, ckpt sha, eval_val zip, B own/inter-z.
