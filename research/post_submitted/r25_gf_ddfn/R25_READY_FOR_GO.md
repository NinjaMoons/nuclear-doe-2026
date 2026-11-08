# R25 — ready for GO (NIH-POST-R25-GF-DDFN-037, POST-SUBMISSION research)

**Status:** `WAITING_FOR_REVIEW`. No training beyond the 2×100 smoke and a 2×20 pipeline dry test (VAL was not evaluated in either).
**Code:** `research/post-submission-r25-gf-ddfn`. The pilot pipeline was QA-fixed at `d06b4b3`; later commits add only diagnostics and documentation.

## Review blockers (all external)

1. Explicit `SCIENCE_GATE=GO` or `NO_GO` from @ggmaster913 (#49, NIH-POST-R25-SCIENCE-REVIEW-038), or from the fallback @playerr456 (#50).
2. Nothing else. The coordinator's QA corrections (#51/6719422836) were done in `d06b4b3` (#51/6719454805).

## One-command launch (Git Bash on DESKTOP-T1K698V, only after GO)

```
bash /c/nuclear-doe-operator/r25/research/post_submitted/r25_gf_ddfn/run_r25_pilot.sh > /c/ndoe-out/r25_p2.run.log 2>&1
```

## Locked settings (no VAL-based choices)

| item | value |
|---|---|
| init | frozen V2 `19b05dad…` (SHA checked) |
| arms | control = V2 fine-tune; treatment = V2 + zero-init dPsi |
| optimizer | Adam, lr 1e-3, fresh state in each arm |
| training | batch 4, R3 objective, AMP off |
| data and order | TRAIN512 (manifest 76d2474d); sampler `Generator(30000+2026)` gives the same quartet order in both arms |
| budget | 6000 updates per arm |
| evaluation | one endpoint `step_006000.pt` per arm, VAL16 evaluated once |

Capacity is **not** matched: the arms have 452,481 vs 509,669 parameters. Only the number of updates is matched.

## Safeguards

- `set -euo pipefail` and an ERR trap. Any failure prints `RUN_FAILED line N` and exits nonzero.
- **Preflight, before anything runs:**
  - Git Bash/MINGW is required;
  - the output dir `C:/ndoe-out/r25_p2` must not exist (it is never reused);
  - the V2 SHA must match;
  - there must be no uncommitted R25 source;
  - no python process may be on the GPU.
- **Order of work:** train both arms with `--no-eval`, then `PROVENANCE_OK`, then `eval_r25.py` with `--expect-sha`, then `compare_r25.py`.
- **`PROVENANCE_OK` checks:**
  - each checkpoint SHA equals its summary and the file is nonempty;
  - updates, seed and all512 are as locked;
  - reload diff is 0;
  - V2 SHA and training-script SHA are equal across the arms.
- No TEST: the train+val loader is `pilot.native`. Both training and eval go through it.

## Expected outputs (`C:/ndoe-out/r25_p2/`)

- `source.txt`, `source_sha256.txt`
- `{control,dpsi}/step_006000.pt` and `r25_summary.json` (train time, peak VRAM, checkpoint SHA, step log)
- `ckpts.txt`
- `eval_{control,dpsi}/` with the official export: DOE.png L8 and four L8 amplitude PNGs per VAL quartet
- `compare/compare.json`:
  - PRIMARY: native P, SSIM and worst plane, with a paired quartet-level bootstrap of the 95% CI;
  - DIAGNOSTIC: own-argmax, registered and LP inter-z, dark fraction, speckle, DOE roughness.
- `compare/sheet_{B64_012,B64_016,A64_060}.png` (raw images: TARGET, then rec + DOE for V2, control, treatment and the diagnostic Adam oracle)

Training takes about 6 minutes of GPU time (measured train step p50: 15.5 ms control, 22.7 ms dpsi), plus about 2 minutes of evaluation.

## Afterwards

Write `R25_PHASE2_RESULTS.md` and `result.json` and post the verdict to #51 (`NO_GO` / `INCONCLUSIVE` / `PROMISING_POST_SUBMISSION`), following the gates in #51/6719281299.
