# Contest organizer's answers and precise final demo contract · 08 Nov 2026

**Status:** based on verbatim replies forwarded by Pavel and archived in [GitHub #7](https://github.com/NinjaMoons/nuclear-doe-2026/issues/7). This is a team compliance interpretation, **not an official organizer certification of our exact architecture**. Do not present unknown answers as confirmations.

## Exact relevant organizer statements

1. Dmitry: «Строго однопроходную не обязательно, можно и несколько. Гибридные подходы тоже можно, соответственно задача не улучшить нейросетевой обработкой существующий метод, а использовать нс для синтеза в первую очередь и при необходимости подкреплять ее работу физикой». [Context and distinction](https://github.com/NinjaMoons/nuclear-doe-2026/issues/7#issuecomment-6709711373).
2. When asked if trained NN DOE prediction followed by 5–20 **per-new-input Adam steps** through differentiable Fresnel was acceptable, Dmitry replied: «Тут у вас результат полноценный выдаётся не нс, а шагами, а нс им помогает». [Exact exchange and operational STOP](https://github.com/NinjaMoons/nuclear-doe-2026/issues/7#issuecomment-6709820195).
3. Another earlier answer: «Использование нс для начальной фазы итеративного метода это будет скорее задача оптимизации итеративного метода. Ваша задача состоит именно в нейросетевом синтезе. Физические модели внутри сети, а также использование классических методов для данных остаётся допустимым». (User-provided quote in team communications; do not silently paraphrase into a stronger approval.)

## What we KNOW and what we do NOT know

| Implementation | Current status | Why |
|---|---|---|
| Trained model synthesizes phase mask directly | SUPPORTED | Primary synthesis is neural |
| Network internally uses fixed physics-derived features to condition prediction | CONSISTENT WITH REPLIES, exact feature pipeline not expressly accepted | Dmitry says physics inside network may be used |
| Multi-stage *neural* inference | SUPPORTED IN PRINCIPLE | Strict single-forward not mandatory |
| Per-input NN output → Adam 5–20 steps as final contest result | **NOT ACCEPTABLE** | Dmitry explicitly judged result primarily optimizer-produced |
| Offline optimizer teacher ONLY on own TRAIN input for research/training | SUPPORTED IN PRINCIPLE, verify training data rights | Earlier replies say classics for data allowed, not competition input refinement |
| External WhiteCollars-trained contest model | NOT CLEARED | Rights and domain performance; current selected native-only V2 has wc=null |
| Specific 12ch fixed inverse-Fresnel conditioning architecture | NO EXPLICIT ORGANIZER ACK | Do not invent; can demonstrate 1 NN forward, 0 optimizer steps by code audit |

## Final scientific/engineering deliverable in reality

Four unknown grayscale **amplitude** target PNG `T1..T4` shape 128x128 (L8), wavelength 532nm, DOE pixel pitch 10µm, propagation z=[24.1, 25.3, 26.5, 27.7]mm:
1. preprocessing: 4 raw target channels and 4 deterministic complex inverse-Fresnel backprojections (=8 Re/Im channels) stacked as 12ch STEM U-Net input;
2. learned neural network **one forward** synthesizes ONE phase-only array, encoded as **128x128 L8 `DOE.png`** mapping phase [-π,π] onto uint8 [0,255];
3. independent physical forward model propagates the reloaded phase DOE to 4 z using **single-FFT Fresnel** and emits 4 native-grid amplitude `|U|` L8 reconstructed PNG, NOT intensity `|U|²`. Output native pitch varies per z;
4. per-plane 4 Pearson + 4 **global** SSIM computed from the saved+reloaded uint8 reconstructions and original targets, plus optional identity/ghosting diagnostic matrix;
5. source, neural weights/checkpoint SHA, CLI, README, exact model version and actual readback/hashes.

**What the product cannot honestly claim:** the 4 target shapes are perfectly isolated on independent Group B planes. V2 VAL16 P=.61243/S=.55356 with Group B P=.4516, B own-target argmax 25/32 but high ghosting in median B64:012 and lost z1 in B64:016. Multiple layers often share a visible mixture. Report these failures without hiding. Strong train-only Adam diagnostic (P~.795/B~.653) demonstrates room for better training but **is not contest inference**.

## Verified artifacts

- Current independently checked V2 **asset 721714285**, SHA256 `ffc9d91ebc4977018e9d8625744657025f7c0c09bf410b2a532fd46dd6bef0b0`; V1 asset721351745 and 4ch fallback asset720489303 are protected.
- Matvey [#50](https://github.com/NinjaMoons/nuclear-doe-2026/issues/50) / [PR #52](https://github.com/NinjaMoons/nuclear-doe-2026/pull/52): clean CPU shell smoke and certified raw 4-z defense figures with full manifest, private **asset 721948530**, SHA `62b96b6bfec88cb686b0db44c79ebbea260ebf5c7680c68a5d1a0fce43d06bec`.
- [Scientific independent verification #44](https://github.com/NinjaMoons/nuclear-doe-2026/issues/44), [V2 review #40](https://github.com/NinjaMoons/nuclear-doe-2026/issues/40).
- [External friend independent QA #53](https://github.com/NinjaMoons/nuclear-doe-2026/issues/53): separate red team, not proof of model quality until actual tested results.

## P0 remaining

- Verify the **official upload destination/procedure**, submission format and correct team identity; archive screenshots/receipt and uploaded ZIP SHA. A private GitHub Release ≠ official contest submission.
- If organizer does not answer the exact feature-conditioning question, do NOT speculate; rely on the evidence and conservative classification above, preserve 4ch backup and ask an authorized team submitter to make the final decision with known risk.
- Align slides and source evidence with the actual final checkpoint if a later research candidate passes scientific gates. Honest 4-z raw slides for A64:060 / B64:012 / B64:016 already exist in asset 721948530. Do not regenerate or retouch them.
- Time boundaries Moscow: new research train starts end 02:00; research cutoff 04:00; final package+deck freeze 05:30; aim official upload 07:30–08:00; internal 09:30 and organizer deadline 10:00.
