# R8-RES — phasor-residual head on V2 (pre-registered 2026-11-08 ~00:40 MSK, before any run)
Why: 5 one-factor training knobs are neutral (B75, width, distill, LR recipe, R17); V2 is within ~.005 of the
architecture+R3 ceiling, while the Adam oracle from the V2 phase reaches P .795 / B own 31/32. Grigoriy's G1 test:
oracle target-plane phase roughness 3.0-3.2 vs V2 0.07-0.42 -> the smooth scalar head cannot express the abrupt
phase the separating solutions use. R8 (#35) proposed a 2-channel phasor head but was never executed.
Factor (one): add `phi += atan2(im, re)` with (re, im) = 1x1 conv over [final decoder features ; 8 back-projection
Re/Im channels]; weights 0, bias (1, 0) -> step 0 == V2 exactly (asserted <= 1e-5 rad). Phase-only, one forward.
Treatment: init V2 19b05dad, R3, Adam lr 1e-3, batch 4, sampler randperm(512, gen(30000+seed))[:4], 6000 updates,
ckpt every 2000, seeds 0/1. Matched control = R17 control_s0/s1 (identical init/sampler/steps/optimizer; same code
path except the head). Primary step 6000 seed mean. GO if dP >= +.01, dP_B >= +.005, dSSIM >= 0, B own not lower,
orgsets10 and WC64 not worse; NO_GO if dP <= -.005 or non-finite; else INCONCLUSIVE. Report B inter-z, raw sheets
A64:060/B64:012/B64:016. Packaging note: a GO would need the head class in the package (loader change) -> separate
review before any contest use. Start before 02:00, p50/p90 25/40 min.

## Addendum R8-SCRATCH (pre-registered 2026-11-08 00:40 MSK, after R8-RES INCONCLUSIVE, before running)
Tests the stated mechanism "the smooth V2 basin blocks the phasor head". Same PhasorStemUNet trained from scratch,
phasor weights N(0,.5), bias 0 (singularities at step 0), sampler randperm(512, gen(seed))[:4], R3, lr 1e-3,
batch 4, 12000 updates, seeds 0/1. Control = existing W16 12k runs L_native_s0/s1 (same sampler/steps; only the
head differs). GO as in the plan (dP >= +.01, dB >= +.005 ...); also report phasor-term roughness. If not GO, the
basin explanation is falsified for this head and #43 research is closed.
