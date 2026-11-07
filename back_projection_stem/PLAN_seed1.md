# STEM seed-1 replication — pre-registered before running (2026-11-07 ~15:58 UTC)
Basis: #5/6709246069 STEM_BETTER (seed 0) and #7/6709328037 priority (2) "seed1 slot"; Pavel autonomy #7/6708606480.
Identical to PLAN_paired.md except --seed 1: control (4-ch) and stem (12-ch), both R3 objective, manifest 76d2474d,
lr 1e-3, batch 4, fp32, CPU 8 threads, 1000 updates, eval-every 100, in-process stock VAL evaluation, code c64c945b.
Readouts: step 1000 and rule 6697684744 per arm; P/SSIM, A/B, per z, worst plane; own-target argmax, inter-z (physical
grid), C diag-offdiag. REPLICATED if at step 1000 dP >= +0.01, dSSIM >= -0.005 and B own-argmax(stem) >= control.
Cap: 30 min wall per arm (sleep-proof check by timestamps). One process on this host. No TEST, no production change.
