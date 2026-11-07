# Adam and trained-NN artifact collector (#6)

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

## Trained NN summary and paired NN/Adam comparison (#6B)

All export checks (both methods) additionally require `output_pitches_m` equal to
`CaseConfigV1().plane(i).output_pitch_m(128, 128)` (rtol 1e-12, the exporter's own
tolerance), `planes[i].reconstruction == reconstructed_{i+1:02d}.png`, and a DOE.png
that `load_doe_png` accepts (PNG, L, uint8, 128x128). The four ordered targets and the
four reconstructions must pass the strict `load_png_quartet` (8-bit grayscale IHDR,
mode L, 128x128) even when their hashes are consistent. Nothing is resampled or reordered.

`summarize_run(run_dir)` (CLI `--run-dir DIR --method any`) returns a schema-2 summary
of an Adam or `fourinput_unet` export. For NN it requires checkpoint SHA256/format,
40-hex code commit, contract version, hparams, dtype and nested `training.seed/steps`;
missing or malformed provenance fails. `training_verified` is copied as recorded
(`false` from the inference CLI, `not_recorded` from the training driver export);
independent acceptance of training is never inferred from a checkpoint
(`independent_training_acceptance = not_recorded_by_collector`). Load/inference times
are reported only when the export records them together with a nonempty scope; a value
without a scope is `null` with scope `not_measured`. The driver export runs batched
inference and records neither. `metadata.manifest_sha256` is reported as
`evaluation_manifest_sha256` (the driver writes the manifest of the EVALUATED data);
the export does not record a training-corpus manifest, so
`training_manifest_sha256 = not_recorded_in_export`.

`compare_runs(pairs)` and the CLI compare already exported runs; they rerun nothing:

```powershell
python -m doe_experiments.benchmark --pair - NN_EXPORT\000 ADAM_EXPORT\A64_014 --pair - NN_EXPORT\001 ADAM_EXPORT\A64_015 --output artifacts/paired-NEW.json
```

`SAMPLE_KEY` `-` uses the key recorded in the exports; an explicit key must agree with
any recorded key (the Adam CLI export records none, the inference CLI export neither).
Each pair must have the same sample_key, the same ORDERED four target SHA256
(`ordered target quartet mismatch at plane i`, no sorting), identical contract,
config, native output pitches, PNG/SSIM policies, phase dtype and synthetic flag.
All pairs must share one NN checkpoint/commit and one fixed Adam seed/steps/lr/dtype;
the paired path requires recorded positive integer `steps` and finite positive `lr`
(the legacy Adam API does not). Recorded `split` and evaluation manifest SHA256 must
agree inside a pair and across all pairs of one report; a side that does not record
them (Adam/inference CLI) takes the other side's value, and if neither records them
the report says `not_recorded`. VAL and TEST, or two evaluation manifests, are never
mixed in one report.

Per metric m: `M_i = mean of the 4 plane scores of quartet i` per method and
`delta_i = M_i(NN) - M_i(Adam)`. A quartet with any undefined Pearson has undefined
quartet Pearson and delta; it is listed in `undefined_sample_keys`, never set to 0.
The report has A, B and pooled blocks (n, method means on the same defined pairs,
delta mean/median/min/max, Q1/Q3/IQR with numpy `linear` = type 7, wins/ties/losses),
per-plane means and mean deltas, and the worst pair. ALL is the half-sum of the A and
B means only for equal defined group sizes, otherwise it is the pooled mean and says so.
The unit is the quartet: 4 planes are not 4 independent observations. No CI, p-values,
Wilcoxon or bootstrap. Timings stay per method with their scopes;
`timing_comparison` is `not_comparable` and no speedup is reported.

The collector does not prove either acceptance gate and does not produce real paired
measurements; those require the gates on the integration HEAD and coordinator approval.
