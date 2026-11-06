# Adam artifact collector (#6)

`summarize_adam_run(run_dir)` and its module CLI collect a completed real Adam
export. They verify official configuration, amplitude policies, ordered plane
references and SHA256 integrity of all manifest artifacts. They preserve four
Pearson/SSIM pairs and Pearson defined flags without recalculating metrics,
normalizing images or invoking a propagator. Hashes validate consistency with
the supplied manifest; they do not authenticate the author or validate physics.

The bounded baseline CLI profile is 16 steps, seed17, float32, synthetic targets:

```powershell
python -m doe_experiments.cli adam --targets a.png b.png c.png d.png --output-dir artifacts/adam16-seed17 --steps 16 --seed 17 --dtype float32 --synthetic-test
python -m doe_experiments.benchmark --run-dir artifacts/adam16-seed17 --output artifacts/adam16-seed17-summary.json
python -m pytest tests/experiments/test_benchmark.py -q
```

The four input paths must already contain ordered 128x128 8-bit grayscale PNGs.
Use `--synthetic-test` only for actual synthetic fixtures. Choose new run and
report paths; the collector never overwrites a report. This document specifies
the collection command, not evidence that a particular real run has passed.
The collector's unit fixtures are explicitly artifact test doubles.

The report carries source/artifact/manifest hashes, config, seed, dtype, policies,
synthetic flag and recorded solver/export times. Solver time covers the CLI
solver call; export time includes DOE write/reload, forwards and PNG/metric/raw
export according to its recorded scope. Training, model load, warmup and isolated
IO are null and explicitly not measured. These two measured scopes are not a
speed comparison with a trained model. `trained_neural=false` and learned
comparison `not_available` remain explicit; no checkpoint is fabricated.

Both scientific acceptance gates must pass separately on the same integration
tree before claiming the real baseline accepted:

```powershell
python -m pytest tests/integration/test_fresnel_reference.py -q
python -m pytest tests/integration/test_submission_roundtrip.py -q
```

The reference gate checks physics independently; roundtrip checks actual module
data transfer. Collector unit success proves neither gate. Display policy is
float64 `ratio=A/max(A)` per plane, then `floor(255*ratio+0.5)`; input PNG/255
already denotes amplitude. The collector performs no display conversion.

A trained-model benchmark extension is pending real training, a checkpoint,
provenance and comparable timing/quality measurements on the same quartet.
