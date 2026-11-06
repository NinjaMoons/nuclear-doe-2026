# Minimal v1 CLI and gate (#7)

This implementation uses the actual #14/#3/#4 modules. Missing modules produce
an error; there is no production stub or ASM fallback. The unit tests use
explicit doubles only to verify export orchestration and do not validate physics.

Install requirements according to the main README, including the pinned Pillow.
For four ordered 128x128 grayscale uint8 PNG files:

```powershell
python -m doe_experiments.cli adam --targets a.png b.png c.png d.png --output-dir artifacts/run01 --steps 16 --seed 17 --dtype float32 --synthetic-test
python -m doe_experiments.cli reconstruct --targets a.png b.png c.png d.png --doe artifacts/run01/DOE.png --output-dir artifacts/repeat01 --dtype float32 --synthetic-test
```

Use `--synthetic-test` only for clearly identified test fixtures. Real supplied
objects must have their actual provenance. Always choose a new output directory.
Both runs must use the same dtype for pixel-for-pixel reproducibility; changing
precision is a separate comparison, not a repeat of the same run.

`DOE.png` is reread before four primary forwards; the same decoded phase is used
at every official z. Continuous-phase forwards are supplemental. PNG conversion
is checked against the fixed per-plane max/round-half-up rule in float64; raw
intensities and per-plane maxima/exposures remain available. Primary metrics use
the reloaded reconstruction PNGs and original targets. The manifest is written
last and records hashes, config/output pitches, dependencies, source flags,
method/seed/timing and a reproduction command. Existing runs are never overwritten.

The real gate is intentionally mandatory:

```powershell
python -m pytest tests/integration/test_submission_roundtrip.py -q
```

Until #14/#3/#4 implementations exist, it FAILS with their missing module names.
It is not skipped and this failure must not be presented as a completed baseline.
After they exist it executes the real CLI, independently verifies reconstructed
pixels from the exported DOE, checks all eight metrics against NumPy equations,
and reconstructs again from the saved DOE to check reproducibility. CI runs unit
tests first and then this gate; the coordinator PR cannot be green on unit tests alone.

Neural CLI/checkpoint support is the next stage, after the physical forward and
shared-phase Adam baseline pass. No long training is started by this CLI or gate.
