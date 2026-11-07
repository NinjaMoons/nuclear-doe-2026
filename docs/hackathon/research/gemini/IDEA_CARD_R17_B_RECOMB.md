# IDEA_CARD R17-BRECOMB — independent-pool B-quartet recombination (train-only), @ggmaster913, 07.11 ~23:58 MSK

Status: CODE-READY spec, NOT started. Single factor. Origin: G0 (Gemini) idea, re-scoped by measured data facts below.

## Measured facts (read-only, manifest 76d2474d, CONFIRMED)
| subset | n | within-quartet inter-plane Pearson (mean / mean|.|) | distinct families per quartet |
|---|---|---|---|
| VAL B (B64) | 8 | +.063 / .069 | 2.12 |
| TRAIN B64 | 48 | +.058 / .085 | 2.60 |
| TRAIN B208 relation=independent | 70 | +.033 / .065 | 3.07 |
| TRAIN B208 relation=partial | 69 | +.227 / .248 | 2.90 |
| TRAIN B208 relation=similar | 69 | +.282 / .376 | 1.00 |
=> Gemini's "B planes are independent" holds only for B64 and B208-independent; 2/3 of B208 TRAIN quartets are
correlated, unlike VAL B. Slot statistics are equal across z (mean level .217-.234), so slot-preserving
recombination is in-distribution for the independent pool.

## Hypothesis (HYPOTHESIS)
VAL-B ghosting partly reflects a TRAIN-B prior of correlated planes; adding slot-preserving recombined quartets from
the independent TRAIN pool (118 quartets -> ~118^4 combinations) gives more decorrelated B supervision and improves
B identity/separation. Not a duplicate: B75 (#43) changed A:B ratio, not inter-plane correlation; width, distill,
D4, axial loss, HF basis are different factors.

## Exact diff vs baseline
- Init V2 `19b05dad…083f`; fresh Adam lr 1e-3; batch 4; R3; fp32; stock-equivalent sampler randperm(512, gen(30000+seed))[:4].
- Control: no change. Treatment: each drawn **B** quartet is replaced with prob 0.5 (private generator 50000+seed) by a
  recombined quartet: plane k (k=1..4) taken from slot k of 4 DIFFERENT quartets drawn uniformly from the pool
  {TRAIN B64} ∪ {TRAIN B208 relation=independent}. A quartets untouched. Same idx sequence in both arms.
- 6000 updates (same as #43 distill control, measured ~100 s/arm on RTX 4070 Ti), checkpoints every 2000, seeds 0/1.
- Leakage guard: pool built from manifest split=="train" only; assert no recombined plane SHA in VAL/TEST manifest SHAs.

## Gate (pre-registered)
Primary step 6000 seed mean vs control: GO if dP >= +.01, dP_B >= +.005, B own not lower, SSIM not lower, orgsets10
and WC64 not worse; NO_GO if dP <= -.005 or visual worse; else INCONCLUSIVE. Report also B inter-z by band
(r<=.15, .15-.3, .3-.5 on the z1 physical grid) and #47 sheets A64:060 / B64:012 / B64:016, hot-px/top-px.

## Cost / owner
- Code 20/30 min (p50/p90) + run 4 x ~2 min + eval ~8 min on RTX 4070 Ti => total ~45/60 min. Must START before 02:00 MSK.
- Owner: @MVRto01 desktop GPU after the W32d3 recipe job finishes (one GPU job at a time). Fallback if GPU not free by
  00:45 MSK: @ggmaster913 Mac CPU, seed 0 only (6000 x 0.23 s ≈ 23 min/arm + eval) ≈ 55/70 min.
- Quota: Grigoriy 5h 81% USED at ~23:3x (Pavel screenshot, reset "11:50 PM", TZ unconfirmed); others UNKNOWN.
- Independent review: @ggmaster913 (if run by MVR) — CPU VAL16 replay + drop-in audit.
