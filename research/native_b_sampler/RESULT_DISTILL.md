# Distillation result (PLAN_DISTILL.md) — **DISTILL_NEUTRAL** (negative trend)
Seed mean, step 6000, VAL16 official path. lam 0 (R3 fine-tune control): P .6141 / SSIM .5567 / A .7750 / B .4531 /
worst .5972 / orgsets10 .5161 / WC64 .3799. lam .6: P .6082 / .5554 / A .7632 / B .4531 / worst .5912 / .5072 / .3700.
dP -.0059 [CI -.0117, -.0011]. Train log: lam .6 reduces the student-teacher phase distance only .58 -> .48-.51 while
TRAIN R3 rises .36 -> .37-.39 (lam 0: distance drifts to .69-.74). => the per-quartet Adam phases are not a mapping
this one-forward U-Net can represent; a phase-space target does not transfer. Student runs 97-105 s each.

# Next (pre-registered here, before running): optimisation recipe
Evidence so far: only training length helped (1k -> 4k -> 12k); B75, width (+.007), distillation are neutral; the
net underfits. Arms on the best width W32d3, native TRAIN512, pilot sampler, R3, batch 4, fp32, seeds 0/1,
**24000 updates**, checkpoints every 4000: **K** constant lr 1e-3 vs **C** cosine 1e-3 -> 1e-5.
Primary: step 24000 seed mean. Promotion over V2 19b05dad needs dP >= +.01 vs .6124, dP_B >= +.005, orgsets10 and
WC64 not worse than V2, then CPU CLI + #40-runtime drop-in check. C vs K reported with quartet bootstrap.
