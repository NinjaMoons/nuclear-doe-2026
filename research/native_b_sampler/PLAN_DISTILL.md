# Teacher-guided physics training (train-only teacher) — pre-registered 2026-11-07 ~23:25 MSK, before any student run

Motivation: RESULT_B75_ORACLE.md (Adam-R3 from the STEM phase reaches VAL16 P .795 / B .653 / B own 31/32 with the
same codec — the gap is model-side; the net underfits: TRAIN64 P .620 ~ VAL .612) and RESULT_CAPACITY.md
(CAPACITY_NEUTRAL: width gives only +.007 P). Hypothesis: the R3 physics gradient alone is a poor optimisation
signal; a per-quartet phase target on TRAIN gives a direct, dense signal toward the separating solution.

- Teacher (DONE before this plan, TRAIN512 only): `distill.py teacher`, init 19b05dad, Adam lr .02, 2000 steps on
  N*R3, kept per quartet only if better (512/512 improved; R3 .3597 -> .2187). File
  `teacher_19b05dad_train512.pt` (sha in run_config). No VAL/TEST/WC/orgsets quartet is ever given to the teacher.
- Student: init 19b05dad, fresh Adam lr 1e-3, batch 4, sampler randperm(512, gen(30000+seed))[:4], 6000 updates,
  checkpoints every 1000, seeds 0/1. loss = R3 + lam * mean(1 - cos(phi_s - phi_t)).
- Arms: **lam 0** (matched control = plain R3 fine-tune) vs **lam 0.6** (fixed by the balance rule
  R3_start / dist_start = .3597 / .5831 = .617, measured before training). No other lambda.
- Inference stays ONE forward of the same 12ch STEM (no teacher, no Adam at inference).
- Primary: step 6000, seed mean, lam .6 − lam 0, VAL16 official path. `DISTILL_HELPS` if dP >= +.01, dSSIM >= 0,
  dP_B >= +.005, and dP >= 0 on orgsets10 AND WC holdout64. `DISTILL_HURTS` if dP <= -.01. Else NEUTRAL.
  Also reported vs V2 19b05dad (.6124 / B .4516), B own-target / inter-z, raw sheets A64:060 / B64:012 / B64:016.
