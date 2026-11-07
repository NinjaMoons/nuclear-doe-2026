# Paired CPU research run CONTROL R3 (4-ch) vs STEM R3 (12-ch) — pre-registered (2026-11-07 ~14:25 UTC)

Authorisation: #5/6708046281 (CONDITIONAL GO). START ONLY AFTER `STEM_OPTICS_ACCEPT` from @ggmaster913 and a free
CPU slot; if the gate is not met in time (latest start 15:10 UTC so that both arms end before 16:00 UTC = 19:00 MSK):
HOLD, report only. Never start around the gate.

Code: worktree 49b96041 + back_projection_stem/ (stem.py 88083fa1..., train_wrapper.py, tests 15/15 OK).
Arms (only the stem differs; both through `python -m back_projection_stem.train_wrapper`):
- control: --arm control  (FourInputUNet 4-ch)
- stem:    --arm stem     (BackProjectionStemUNet 12-ch; every layer after the stem shares the control init, tested)
Common: objective R3 = lsq + 0.3*(1-corr) + 0.1*contract (identical to r3_train_wrapper 9e8cd4a9), manifest
76d2474d (corpus 6c6f136b), seed 0, lr 1e-3, batch 4, fp32, CPU 8 threads, eval-every 100, same batch sequence
(same seed -> same sampler), weights-only checkpoints; in-process stock evaluate_checkpoints on VAL only
(official DOE.png uint8 -> reload -> 4 Fresnel -> 4 PNG -> reload -> Pearson/global SSIM).

Health gate (stem arm, 100 updates, separate out-dir, --no-eval path of the driver + in-process eval of step 100):
finite objective/grad_norm at every step, VAL eval at step 100 runs, phase shape [B,1,128,128], targets order z1..z4
as in the manifest, wall time recorded (VRAM N/A on CPU). Fail -> STOP, BLOCKED with the log.
Pass -> run control 1000 then stem 1000 (sequential, one process at a time). Cap 25 min wall per arm.

Readouts (no hyper-parameter change by VAL):
- primary: step 1000 and the checkpoint chosen in each arm by rule 6697684744 (max VAL mean Pearson; within 0.005 ->
  larger worst-plane SSIM, then earlier step); P/SSIM mean, A/B, per z, worst plane;
- separation: own-target matrix C[k,j] = Pearson(recon_k, target_j) per quartet, count argmax_j == k,
  inter-z Pearson of reconstructions on the physical grid (z2..z4 resampled onto z1's centred grid, bilinear,
  align_corners=True, x=(n-64)*pitch_z) vs the same for targets; visual Target / control / stem sheet.
Decision label (descriptive, one seed): STEM_BETTER if dP >= +0.01 and dSSIM >= -0.005 and B own-argmax not lower;
STEM_WORSE if dP <= -0.01 or dSSIM <= -0.01; else INCONCLUSIVE. TEST, production best.pt, fallback: untouched.
