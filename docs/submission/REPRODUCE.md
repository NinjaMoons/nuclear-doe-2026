# Submission reproduction skeleton (#8)

Status: preparation only, not a completed submission.

Required per-quartet bundle: four ordered original targets and their provenance,
one `DOE.png`, four `reconstructed_01..04.png`, `metrics.json`, `raw.npz`,
`manifest.json` and a collected Adam summary. Preserve config/SI output pitches,
source/artifact hashes, amplitude/SSIM policies, seed, dtype, dependency versions,
git SHA and the exact reproduction command. Primary scores are the four pairs
from saved/reloaded reconstructions of the saved/reloaded shared DOE.

Reproduction skeleton with supplied files and a fresh output directory:

```powershell
python -m doe_experiments.cli adam --targets a.png b.png c.png d.png --output-dir artifacts/submission-adam --steps 16 --seed 17 --dtype float32
python -m doe_experiments.benchmark --run-dir artifacts/submission-adam --output artifacts/submission-adam-summary.json
python -m doe_experiments.cli reconstruct --targets a.png b.png c.png d.png --doe artifacts/submission-adam/DOE.png --output-dir artifacts/submission-repeat --dtype float32
python -m pytest tests/integration/test_fresnel_reference.py -q
python -m pytest tests/integration/test_submission_roundtrip.py -q
```

The 16-step command is a bounded baseline profile, not a claimed quality target.
Use `--synthetic-test` for synthetic inputs in both CLI invocations and identify
that evidence explicitly. Collector accepts method Adam only; a reconstruct run
is a repeat check and cannot be relabeled as synthesis or a trained model.

PENDING: real official input quartet/provenance; verified dataset with independent
splits; trained neural checkpoint, training log and evaluation; both full gates
on the final integration tree; final submission contents and actual measured
comparison. A green collector unit suite does not complete these requirements.
