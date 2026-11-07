# NUCLEAR IT HACK — defense slides: editable source outline

**Draft** — not a rendered slide deck. Can be converted into PPTX/Google Slides by Claude Code, or into a Gemini-generated deck if Claude quota unavailable. **Must sync final checkpoint before export.** Current verified technical reference: native-only STEM V2 `19b05dad…083f`, VAL16 P=.61243, global SSIM=.55356. Speaker notes in [speaker_notes_ru.md](speaker_notes_ru.md); asset manifest and caveats in [DEFENSE_INDEX.md](DEFENSE_INDEX.md).

## Slide 1 · One DOE, four different pictures
- Headline: «Одна нейросетевая фазовая маска — четыре амплитудных изображения на разных расстояниях».
- Visual: clean z1→z4 diagram at 24.1,25.3,26.5,27.7 mm, four grayscale target thumbnails. Do not present targets as reconstructions.
- Spoken point: why phase-only and four planes are challenging.

## Slide 2 · Physical / evaluation contract
- 128×128, DOE pixel pitch 10 µm, wavelength 532 nm, phase-only 8-bit DOE PNG.
- Fresnel propagation creates four native-grid **amplitude** `|U|` outputs, not intensity `|U|²`.
- Quantization+PNG round-trip: network phase → DOE.png uint8 → reload → Fresnel → save+reload 4 reconstructed PNG → Pearson + global SSIM per plane.
- Visual: input→DOE→4 Fresnel outputs. Distinguish physical output pitch across z, about 10.02–11.51 µm.

## Slide 3 · Our neural architecture
- **4 targets + 8 analytical Re/Im inverse-Fresnel features = 12 channels** → trained STEM U-Net → ONE neural forward → one DOE.
- Fixed physics prior as prep of input features, not iterative post-inference solver.
- No per-test-case GS/Adam or optimization after network.
- Visual: simple blocks (4 target PNG, 4 inverse Fresnel complex pairs, 12-channel NN, phase).
- Footnote: **organizer-specific confirmation of this exact inverse-Fresnel features scheme pending** until written evidence arrives.

## Slide 4 · Baseline → v1 → v2: reproducible progress
| | 4ch fallback | STEM v1 s0step900 | STEM v2 native s0step12000 |
|---|---:|---:|---:|
| Pearson VAL16 | 0.4181 | 0.5778 | **0.6124** |
| Global SSIM VAL16 | 0.3215 | 0.5067 | **0.5536** |
| Group B Pearson | 0.286 | 0.4257 | **0.4516** |
- 16 frozen validation quartets (8 A/8 B). Not 64 independent random test samples.
- V2 improves v1 on all 16/16 VAL quartets and on 10/10 preexisting hand-made evaluation quartets; OOD caveat.
- Dataset native TRAIN512. V2 checkpoint has `wc=null` (no external WhiteCollars samples).
- Source receipts: [#41](https://github.com/NinjaMoons/nuclear-doe-2026/issues/41), [#40](https://github.com/NinjaMoons/nuclear-doe-2026/issues/40).
- If a newer checkpoint is promoted: replace complete table, links and independent audit, NEVER just slide headline.

## Slide 5 · Actual images: median Group A
- Four z columns; rows TARGETS / 4ch fallback / v1 STEM / final selected version (currently v2).
- Use exactly official raw reconstructed uint8 files and identical display procedure; source `A64:060` or another fixed, documented representative.
- Caption: sample selection method, z, source model SHA and source asset ID.

## Slide 6 · Group B: progress AND remaining failure
- Top: median B64:012 (own-target may score 4/4 while all four reconstructions show mixed forms).
- Bottom: hard B64:016 (notably missing target z1).
- Labels TARGETS / selected v1 or fallback / final version; no retouch, per-plane gamma or contrast enhancements.
- Evidence: own-target 4×4, B Pearson, B own count, where ghosting occurs. Do NOT claim four clean isolated images when they are overlaid.
- Optional blind test from `#47`: randomize labels but keep key separate, clearly mark descriptive only.

## Slide 7 · Scientific insight and limitations
- Genuine performance gain from learned conditioning and sufficient training (e.g. v2 12000 vs v1 900), independently reproduced.
- Large remaining gap between amortized one-forward inference and **training-only diagnostic** per-sample Adam oracle; e.g. P .7952, B P .6526 on diagnostic set. **Teacher is NOT a valid contest model/inference**.
- Low-frequency axial crosstalk and model underfitting interact; avoid claiming a fundamental impossibility of four separate planes.
- Failed ideas tested honestly: D4 augmentation, R15 axial-loss, B75 oversampling, random HF/carrier, external WC mismatched distribution. Negative results support engineering rigor; do not overfill slide.

## Slide 8 · Reproducibility, deployment and outcome
- 1-command inference with frozen source+checkpoint; returns DOE.png, four reconstructed PNG, four Pearson, four SSIM.
- V2 package private asset `721714285`, SHA `ffc9d91ebc4977018e9d8625744657025f7c0c09bf410b2a532fd46dd6bef0b0` (check actual selected final ZIP).
- Independent Mac clean environment + 20/20 test receipt; exact verified links in [#40](https://github.com/NinjaMoons/nuclear-doe-2026/issues/40).
- End with 1 clear sentence: single-forward neural phase synthesis is reproducible, quality improves, B ghosting remains an engineering frontier.
- **Do not show a submission receipt icon until one exists.**

## Slide creation/asset checklist

- Before rendering: verify current chosen final SHA and schema in #8; verify true raw 4-plane images and chart inputs in private release with full SHA. Change deck together with model or freeze unchanged.
- Design: dark-on-light or light-on-dark, very large image panels and clear four-z labels, avoid text-dense slides. Each plot a reproducible source script/CSV; footnotes cite GitHub issue/asset and sample IDs.
- Write generated PPTX (or editable source), SVG/PNG diagrams, image/source manifest, and `speaker_notes_ru.md` to this isolated git branch, large blobs in private GitHub Releases. Use only actual verified paths.
