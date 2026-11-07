# WhiteCollars external TRAIN corpus — audit + adapter (#41, RESEARCH ONLY)

Status: prepared and tested on synthetic fixtures only. The real archive
`WhiteCollars_gray_train4096.tar.gz` (sha256 `f58db14e…c29fc521`, 49 103 635 B) is NOT on this host and the author's
permission for training use is NOT confirmed, so nothing here has been run on WhiteCollars data and no model has
been trained on it.

- `external_dataset_audit.py` — @ggmaster913's #14 auditor, byte-identical to sha256 `863171ca…` (#14/6709043764).
- `wc_audit.py` — quartet-level extension (z-order parsing without guessing, split mapping, identical planes,
  duplicate / z-permuted quartets, overlap with our TRAIN/VAL quartets by decoded pixels and with TEST by manifest
  file sha only, dHash hits re-checked by pixel MAE/Pearson, distribution stats vs our TRAIN A/B, deterministic
  64-quartet internal holdout). Writes `WC_AUDIT.json`, `WC_MAPPING.json`, `WC_MAPPING.md`.
  `training_allowed` stays false until `--rights-evidence` names a file with the author's written permission.
- `adapter.py` — `WhiteCollarsTrain(archive, WC_MAPPING.json)` → float32 `[B,4,1,128,128]` = uint8/255, identical
  to `load_png_quartet` on the same bytes; verifies archive and per-file sha; never exposes non-train splits;
  holdout only with `role="holdout"`.

```
python -I research/whitecollars_dataset/wc_audit.py WC.tar.gz <our_corpus_dir> <new_out_dir> --expect-sha256 f58db14ea2fbe4235bfe4ea21855a45d52b2d506700f0c0f82ae16a3e29fc521 [--rights-evidence permission.txt]
python -m pytest research/whitecollars_dataset/tests -q
```
Our corpus dir needs `manifest_expanded.json` (76d2474d…) and TRAIN/VAL images; TEST images are never opened and may
be absent.
