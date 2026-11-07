# REPRO_CHECKLIST — STEM contest packages (v2 primary, v1, 4ch fallback)

Owner: @playerr456 (#50, #40). Written 2026-11-07 ~21:00 UTC (00:00 MSK). Evidence only — nothing here is an official submission.
Every PASS below links to an issue comment with hashes. "UNTESTED" means nobody has run that variant yet.

## 0. Artifacts (private release `data-4b-v1`; verify bytes before use)
| package | asset | bytes | SHA256 | checkpoint SHA256 | VAL16 P / SSIM |
|---|---|---:|---|---|---|
| **v2 STEM native12k s0** | 721714285 | 6 109 554 | `ffc9d91ebc4977018e9d8625744657025f7c0c09bf410b2a532fd46dd6bef0b0` | `19b05dad34601a8b5ea4c78e7cd6904b0c639a218f8fbcdeabc2afbb5d6f083f` | 0.6124 / 0.5536 |
| v1 STEM s0 step900 | 721351745 | 3 406 898 | `347eecc254b3022716a56d5a3bedf5e6504b9b7f64806dcf43256f863ed84bff` | `569e170065cca160eb6ce64a0707d81e5e016857b0962498c156cba80f6d3b7d` | 0.5778 / 0.5067 |
| v1 LF checksum file | 721422163 | 9 554 | `6aee29f1d667978ac3d2dc926dddb4bd8a634cdb74e7e8a30d7268d052a82778` | — | — |
| 4ch fallback kit (laptop) | 720489303 | 65 045 200 | `a358e865a659e1fc73489c7bb2f80de0ef18d55b60355de12b4c4b1282f4ed12` | `ba699b7464ca448dd367399a5509914468c23c0d1990bb3771692f4ad4fb16a2` | 0.4181 / 0.3215 |

The `src/` of v1 and v2 is byte-identical: runtime `6785eca6e5c7f94a914214c1aac5bd5984792824` (65 files = git blobs, LF) plus `back_projection_stem/` @ `1790a37b`. Only the checkpoint differs.

## 1. Environment (Python 3.12, CPU)
**Windows PowerShell**
```
py -3.12 -m venv venv            # without the Python Launcher: C:\path\to\python3.12.exe -m venv venv
.\venv\Scripts\python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
.\venv\Scripts\python -m pip install numpy==2.3.4 Pillow==13.0.0
```
**Linux / macOS (bash/zsh)**
```
python3.12 -m venv venv
venv/bin/python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
venv/bin/python -m pip install numpy==2.3.4 Pillow==13.0.0
```
Expected `pip freeze`: `torch==2.9.0+cpu`, `numpy==2.3.4`, `pillow==13.0.0` (plus filelock, fsspec, Jinja2, MarkupSafe, mpmath, networkx, sympy, typing_extensions, setuptools).

## 2. Integrity (from the unpacked package root)
- **v2:** `sha256sum -c SHA256SUMS.txt` → 94/94 OK. The file uses LF.
  - Windows without coreutils: `Get-FileHash -Algorithm SHA256 <file>` for spot checks.
- **v1:** the inner `SHA256SUMS.txt` has CRLF. Use one of:
  - `sha256sum -c stem-candidate-6785eca-s0step900.SHA256SUMS.txt` (asset 721422163), or
  - `tr -d '\r' < SHA256SUMS.txt | sha256sum -c` → 96/96 OK.

## 3. Tests (from `src/`)
| shell | set variables | PASS? |
|---|---|---|
| **PowerShell** | `$env:BPS_STEM_PT="..\model\<ckpt>.pt"; $env:BPS_INPUTS="..\examples\synthetic_5F"` | **correct** |
| PowerShell | `set BPS_STEM_PT=...` | **WRONG.** `set` is `Set-Variable`, no environment variable is created, and the tests are silently **skipped** (measured: `test_infer` "Ran 3, skipped=3"). |
| cmd.exe | `set BPS_STEM_PT=..\model\<ckpt>.pt` and `set BPS_INPUTS=..\examples\synthetic_5F` | correct |
| bash/zsh | `export BPS_STEM_PT=../model/<ckpt>.pt BPS_INPUTS=../examples/synthetic_5F` | correct |
```
python -m unittest back_projection_stem.test_stem back_projection_stem.test_infer back_projection_stem.test_package
```
Expected: **20 tests OK**. 3 are skipped unless `BPS_BEST_PT` points to the 4ch `best.pt`; with it, all 20 run.

## 4. One-command inference (from `src/`; NEW output dir each time)
```
python -m back_projection_stem.infer_neural_stem --checkpoint ../model/<ckpt>.pt --targets <z1.png> <z2.png> <z3.png> <z4.png> --output-dir <NEW_DIR>
```
- **Inputs:** 4 PNG, 128×128, grayscale L8, order z = 24.1 / 25.3 / 26.5 / 27.7 mm. Quote paths with spaces.
- **Flag:** `--synthetic-test` only for check inputs.
- **Outputs:** `DOE.png` (uint8, 0 = −π, 255 = +π), `reconstructed_01..04.png`, `metrics.json` (per-plane Pearson and global SSIM), `manifest.json`, `raw.npz`, `targets/`.
- **Archive build:** `git_sha` is `null` in the manifest — provenance comes from MANIFEST.json and SHA256SUMS, not from git.
- **Fallback 4ch:** stock `python -m doe_experiments.cli neural --checkpoint <best.pt> --targets ... --output-dir NEW`.

## 5. Reference check (VAL A64:006 shipped in `examples/`)
- **v2:** metrics must equal `evidence/expected/val_A64-006_cpu_reference_metrics.json`. On the builder CPU the DOE is byte-identical (`6d926e6e…e521`).
- **Another machine:** compare **metrics with tolerance 1e-4**, not output SHA. Single DOE pixels can differ by ±1 code (MVR, #40/6710385464).
  - Measured cross-host: v2 Windows CPU vs RTX 4070 Ti — max |Δ| 1.1e-6; v1 Windows vs Mac M1 — DOE identical (#44).

## 6. PASS / FAIL evidence
| check | v2 | v1 | 4ch fallback |
|---|---|---|---|
| asset re-download SHA + `testzip` | PASS (#40/6712031304) | PASS (#40/6710485713) | PASS (#8/6706172798, #40/6706293621) |
| clean venv per §1 (Windows) | PASS (#40/6712031304; 294 s install) | PASS (#40/6711718409) | — (uses same runtime) |
| checksums §2 | PASS 94/94 | PASS 96/96 (LF file / `tr`) | PASS 41/41 |
| tests §3 PowerShell `$env:` | **PASS** 20 (3 skip w/o best.pt) | PASS 20/20 | n/a |
| tests §3 cmd.exe `set` | **PASS** (`test_infer` 3 run, 1 skip w/o best.pt) | UNTESTED (same code) | n/a |
| tests §3 macOS bash/zsh | PASS (Grigory #40/6712086695) | PASS (Grigory #44/6710672576) | n/a |
| tests §3 Linux | UNTESTED (same POSIX commands as macOS) | UNTESTED | n/a |
| one-command CLI §4 + format checks | PASS (acceptance receipts asset 721714985) | PASS (receipts asset 721351995) | PASS (VAL16 = 0.4181/0.3215, #40/6711718409) |
| reference §5 | PASS (byte-identical on builder; 1.1e-6 cross-host) | PASS (byte-identical; Mac identical) | PASS (= #5F receipt) |
| one neural forward / no optimizer at inference | PASS (`test_package`: 1 forward, 4 inverse Fresnel, 0 optimizer/backward) | PASS | PASS (stock CLI) |
| new synthetic OOD quartets | PASS format; quality #40/6712042632 | PASS format; quality #40/6711454958 | PASS format |

## 7. Known pitfalls (read before the final run)
1. PowerShell `set` does not set environment variables. Use `$env:VAR="…"`.
2. The v1 inner checksum file has CRLF. Use the LF asset or `tr -d '\r'`.
3. The output dir must be new; existing results are never overwritten.
4. Cross-host DOE bytes may differ by 1 code in single pixels. Judge by metrics (1e-4), not SHA.
5. Input order is z1 → z4. Images that are not 128×128 L8 → **STOP**, do not resample.
6. Do not open frozen TEST. These packages contain no TEST files.
7. A private GitHub release is **not** an official submission. The receipt comes from the organizer channel (#8).
