# Addendum (pre-registered 2026-11-07 ~21:45 MSK, after the 4000-update pilot, before any longer run)

Pilot result (ANALYSIS_4000.md): WC_NEUTRAL (dP_VAL16 -0.0065, CI excludes 0; orgsets10 -0.016) -> WhiteCollars is NOT
used further. Positive, replicated over 2 seeds: native STEM+R3 trained 4000 updates beats the #40 candidate
(569e1700) on VAL16 and orgsets10; CPU one-forward CLI reproduces the GPU metrics to 1e-4.

Extension (only factor = number of updates): arm A native exactly as in PLAN_WC_PILOT.md, seeds 0 and 1,
**12000 updates**, checkpoints every 2000. Primary readout = step 12000 (seed mean) vs step 4000 of the pilot.
`LONGER_HELPS` if dP_VAL16 >= +0.005 and dSSIM >= 0 and orgsets10 dP >= 0; otherwise the step-4000 checkpoints stay
the research candidates. No selection among intermediate checkpoints by VAL (reported only). No TEST. No WC.
Any candidate stays research-only; replacing the #40 package is Pavel's decision.
