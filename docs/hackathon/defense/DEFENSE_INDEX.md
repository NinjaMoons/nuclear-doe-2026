# Nuclear IT Hack 2026 — evidence-first defense hub

> **Canonical defense INDEX (living draft).** Written 07 Nov 2026, night shift, on branch `ops/night-research-defense-20261107`.
> Source of truth for empirical claims = exact artifacts + hashes + issue comments, not a slide author's recollection. Do not claim official submission/organizer STEM approval without evidence.

## Executive facts, verified

**Contest contract:** input FOUR grayscale **amplitude** 128×128 targets at z=24.1,25.3,26.5,27.7 mm; λ=532 nm, DOE pixel pitch 10 µm; ONE 8-bit phase-only DOE; 4 native-grid single-FFT Fresnel `|U|` amplitude reconstructions; per-plane Pearson and *global* SSIM from **quantized DOE.png reloaded** and **saved reconstructed PNG reloaded**. Our submitted solution uses one trained neural forward to produce the DOE; the organizer allows neural approaches but rejected per-input Adam/GS iterative refinement after the network. Exact STEM use of deterministic inverse Fresnel features still awaiting organizer-specific acknowledgement.

| Model | Frozen checkpoint / package | VAL16 Pearson | VAL16 SSIM | A Pearson | B Pearson | B own targets | Status |
|---|---|---:|---:|---:|---:|---:|---|
| pure-NN 4ch fallback | asset 720489303 | .4181 | .3215 | .551 | .286 | **8/32** | Protected |
| STEM v1 s0 step900 | asset 721351745, checkpoint `569e1700…3b7d` | .5778 | .5067 | .7298 | .4257 | 24/32 | Protected, independently reviewed |
| **STEM v2 s0 step12000 native-only** | asset **721714285**, checkpoint `19b05dad…083f` | **.61243** | **.55356** | **.7733** | **.4516** | **25/32** | Separate ZIP, independently audited, not necessarily submitted |

V2 package SHA256: `ffc9d91ebc4977018e9d8625744657025f7c0c09bf410b2a532fd46dd6bef0b0`.
V1 package SHA256: `347eecc254b3022716a56d5a3bedf5e6504b9b7f64806dcf43256f863ed84bff`.
Native V2 training asset `721613794`; independent cross-host audit asset `721632005`; independent V2 package audit asset `721727913`.
Refs: [submission #8](https://github.com/NinjaMoons/nuclear-doe-2026/issues/8), [packaging #40](https://github.com/NinjaMoons/nuclear-doe-2026/issues/40), [training #41](https://github.com/NinjaMoons/nuclear-doe-2026/issues/41), [visual QC #47](https://github.com/NinjaMoons/nuclear-doe-2026/issues/47).

**Performance caveats:** v2 beats v1 16/16 frozen VAL quartets and 10/10 hand-made OOD targets by Pearson; 10 OOD sets were for evaluation, not training/checkpoint selection. Small VAL16 is limited evidence of generalization. Group B remains poor: B64:012 visually blends all four figures, and B64:016 loses z1. B own 25/32 does not mean 25 optically isolated reconstructions. Show at least one failure case honestly. Do not label Adam-oracle performance as contest inference performance.

## Defense asset locations

Binary checkpoints, image sheets, and big source archives **stay in the repository's PRIVATE GitHub Releases**, with immutable asset IDs, byte length and full SHA256. This index and the small presentation artifacts (Markdown, diagrams-as-source, CSV/JSON) are tracked in git. No vague references to files on only someone's desktop.

- [V2 separate submission-ready candidate, #40](https://github.com/NinjaMoons/nuclear-doe-2026/issues/40): private asset `721714285`, SHA `ffc9d91e…dd6bef0b0`.
- [V1 protected, #40](https://github.com/NinjaMoons/nuclear-doe-2026/issues/40): private asset `721351745`.
- [4ch protected fallback, #8](https://github.com/NinjaMoons/nuclear-doe-2026/issues/8): private asset `720489303`.
- [A/B/hard-B visual QC, #47](https://github.com/NinjaMoons/nuclear-doe-2026/issues/47): asset `721615098`, raw four-plane sheets and blind test, SHA `8e0c341a…dbb49d4b51`.
- [Frozen VAL exports #40](https://github.com/NinjaMoons/nuclear-doe-2026/issues/40): asset `721640409`, SHA `32ffca32…5f8cfa20e`.
- [V2 independent review, #40](https://github.com/NinjaMoons/nuclear-doe-2026/issues/40): asset `721727913`.
- [V2 native-only long training and contact sheets, #41](https://github.com/NinjaMoons/nuclear-doe-2026/issues/41): asset `721613794`.
- New research assets: owner appends exact artifact IDs, URLs, hashes, summary and group-specific results here **only after actual upload/readback**.

## Slide-ready narrative (suggested 7–8 slides)

1. **Problem:** one DOE to synthesize four different amplitude goals on four z planes; include geometry diagram with correctly labeled native output pitches.
2. **Optical forward model:** single-FFT Fresnel, phase-only DOE, quantized uint8 contract, amplitude vs intensity, 4 measured outputs.
3. **Baseline & failure:** 4ch pure NN, baseline Pearson .4181 and SSIM .3215; compare raw example A/B only with correct checkpoint.
4. **Neural approach:** deterministic back-projection Re/Im channels (4 targets+8 physics features) → trained STEM U-Net → ONE phase output in ONE network forward. Don't falsely claim organizer's exact architecture acceptance.
5. **Measured improvement:** honest 4ch/v1/v2 table, VAL16 and A/B. Explicitly label test set provenance, n=16 quartets, don't report 64 planes as independent samples.
6. **Raw reconstructed images:** four columns for z1–z4, rows TARGET/4ch/v1/v2 on predeclared A median, B median, B worst; show original unprocessed official PNG with identical display semantics and source/checkpoint hashes.
7. **Scientific analysis:** B ghosting limitation, B own-target matrix and failure analysis; teacher with per-input optimizer is a diagnostic upper bound ONLY, NOT permitted final inference. Note confidence limits and that low-frequency phase sharing makes cross-plane leakage difficult; avoid claiming ghosting is physically impossible.
8. **Submission/repro:** exact checkpoint SHA, CLI readme, isolated package cold-start/20 tests, hashes, limitation and demo. Include actual submission receipt only after verified.

## Required media / presentation outputs (not yet claimed completed)

Owner `@NinjaMoons` via #47: make small files on a dedicated git branch under `docs/hackathon/defense/`:
- `slides_outline.md`: 7–8-slide thesis and provenance footnotes;
- `speaker_notes_ru.md`: 3–5 minute narrative and 90-second emergency pitch;
- `metrics.csv` or `metrics.json`: exact model/seed/step and source paths, no fake precision;
- `references.md`: scientific claims and links, explicit assumptions;
- `figures_manifest.json`: for each image: quartet ID (A64:060/B64:012/B64:016), model checkpoint SHA, official PNG paths/asset IDs, raw SHA, z/pitch, original vs displayed checksum;
- a source-editable slide deck using available presentation tooling and saved in a GitHub branch or release with checksum. Never confuse a GitHub issue/asset link with a file physically present in the git checkout.

**If Claude credits run out:** Gemini may create slides using this index, exact copied source/figures + SHA and current model stats; labels and caveats must survive. Avoid screenshots of tables where machine-readable source exists.

## Release / acceptance gate

- A model trained on unlicensed WhiteCollars is not the selected native V2; keep source rights traceability.
- Latest selected checkpoint for the defense must equal the final release checkpoint; **if training capacity experiment wins, update ALL corresponding data, diagrams, captions and sources, not just the number**.
- Verify all artifact URLs and hashes, every chart and image is sourced, and there is no hand-made contrast or denoising of the reconstructions.
- Submission channel and real receipt are **PENDING** until independently verified. Team internal 08 Nov 09:30 MSK; organizer email says absolute deadline 10:00 with Stop Coding 09:00–09:30. Aim earlier. No claims that uploading to a private GitHub draft constitutes official contest submission.


## 07 Nov night: CPU host and distilled-student preparation

- Matvey laptop: Intel i5-13420H (8 cores/12 threads), 15.7 GiB RAM (4.4 GiB free at audit), Intel UHD graphics, **no NVIDIA/CUDA**; PyTorch 2.9.0+cpu. Measured STEM step 0.25–0.6s, VAL export adds ~3min. Evidence [#40 host report](https://github.com/NinjaMoons/nuclear-doe-2026/issues/40). Do not run long training on this host while useful CPU parallel tasks remain.
- Matvey's R16 teacher-student pilot is PRE-REGISTERED, code+6 unit tests PASS, **NOT TRAINED / NOT PROMOTED**. Code branch `research/r16-teacher-student-prep` @ `cce75bc2`, review via [#49](https://github.com/NinjaMoons/nuclear-doe-2026/issues/49). Diagnostic TRAIN-only teacher50 improves on four small train examples; this is NOT independent proof that student improves, nor a valid contest inference algorithm.
- Use idle CPU for R16 static tests, optics/metric QA, original contact-sheet verification, provenance/manifest and editable slide-source generation while Sanya's separate RTX4070Ti runs width/depth research. If Matvey's HOME DESKTOP becomes available remotely, first report actual GPU model/VRAM, torch/CUDA, jobs and speed before adding any parallel train job.
- B own-target for 4ch = **8/32** on frozen VAL (not unknown); STEM v1 B own = 24/32 and V2 = 25/32, with ghosting still substantial on B. Source [#47 QA addendum](https://github.com/NinjaMoons/nuclear-doe-2026/issues/47).
- Claude quota screenshots given by Pavel around 23:16 MSK: Matvey 5h **43% likely remaining** resetting Sun 02:30 MSK, weekly **53% likely remaining** resetting Friday 05:00; Sanya session **34% USED** resetting Sun 00:50 MSK, weekly 16% USED resetting Friday 21:00. Screenshot interpretation may differ by product; treat as provisional and ask agent to check live UI before planning major call-heavy job. Do not store account secrets.
