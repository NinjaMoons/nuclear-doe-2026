# Minimal v1.1-amplitude CLI and two acceptance gates (#7)

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
is checked against the fixed per-plane AMPLITUDE max/round-half-up rule in float64; PRIMARY amplitudes and per-plane maxima/exposures remain available; true squared
intensities are separate diagnostic arrays. Primary metrics use
the reloaded reconstruction PNGs and original targets. The manifest is written
last and records hashes, config/output pitches, dependencies, source flags,
method/seed/timing and a reproduction command. Existing runs are never overwritten.

Both gates are intentionally mandatory:

```powershell
python -m pytest tests/integration/test_fresnel_reference.py -q
python -m pytest tests/integration/test_submission_roundtrip.py -q
```

Until #14/#3/#4 implementations exist, it FAILS with their missing module names.
It is not skipped and this failure must not be presented as a completed baseline.
After they exist the roundtrip executes the real CLI and verifies reconstructed
pixels from the exported DOE, checks all eight metrics against NumPy equations,
and reconstructs again from the saved DOE to check reproducibility. Calling the
same production propagator here verifies dataflow, NOT independent physics.
The separate reference gate compares complex fields and amplitude-loss gradients
against a direct integral-kernel quadrature without FFT or production grid/pitch
helpers, including odd/anisotropic grids and four official distances. Its fault
checks reject scale/conjugation/orientation/shift errors hidden by a shared forward.
CI runs unit tests and both mandatory steps; unit success alone cannot accept this PR.

Neural CLI/checkpoint support is the next stage, after the physical forward and
shared-phase Adam baseline pass. No long training is started by this CLI or gate.

Contract v1.1-amplitude uses abs(U), not abs(U)**2, for labels, loss and submitted PNGs.
Input PNG/255 already denotes target amplitude: never apply an additional square root.
