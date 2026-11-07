# Промпты для двух сессий Gemini (#49) — вставлять ЦЕЛИКОМ, каждый блок в свою сессию

Источник: ops/night-research-defense-20261107:docs/hackathon/research/GEMINI_TO_CLAUDE.md. После ответа сохрани ПОЛНЫЙ ответ без правок в файл (G1_answer.md / G2_answer.md на рабочем столе) и укажи модель Gemini и время.

## Сессия 1 — G1 (оптика)

```
LIVE FACTS:

One phase-only DOE 128² 8-bit, 4 distinct grayscale **amplitude** targets 128², z=24.1/25.3/26.5/27.7mm, λ=532nm, input pitch10µm. Single-FFT official Fresnel, native output pitch changes with z. Metrics Pearson and **global** SSIM on reloaded uint8 recon PNG. Inference = one trained neural forward; per-input iterative Adam/GS disallowed. Architecture 12ch = 4 targets + 4 complex inverse-Fresnel Re/Im (8 channels) feeding trainable STEM U-Net, original R3 objective.

Current strongest proven native-only model is v2 seed0 step12000 P=.61243 SSIM=.55356 A_P=.7733 B_P=.4516, B own 25/32. Asset 721714285, independently checked on Mac. B median has serious ghosting despite own 4/4. Earlier step900 P=.5778 S=.5067. B75 oversampling 25/75 neutral; WhiteCollars training harms our domain and rights not cleared; D4 c64 exact but model effect neutral; R15 +0.05 axial-loss neutral; R13 quadratic carrier NO-GO; random HF-basis not helpful. Sanya currently running wider/deeper network tests W32d3 W64d3 W32d4, 2 seeds, 12k steps; follow-on Adam-teacher distillation contemplated only if capacity not beneficial, under train-only constraints. No new Gemini task should duplicate GPU width jobs.

Adam 2000-step *per sample* from STEM initialization can reach P=.7952, B_P=.6526 on diagnostics, **NOT competition inference**. It suggests learnable gap; do not treat as neural model performance or guarantee. Geometric axial crosstalk is physically complex: do not assert four planes are fundamentally inseparable.

TASK:
You are a skeptical Fourier optics physicist. Explain and quantify observed axial ghosting for our exact native-grid single-FFT Fresnel at four z. Distinguish mathematically limited low-frequency selectivity from recoverable model error (Adam teacher). Consider physical output scaling, Fresnel transfer, phase-only constraint, and finite aperture. Give a scientifically defensible 2-paragraph explanation and a slide-ready diagram proposal, explicitly mark approximations. Suggest no more than TWO unique, falsifiable read-only tests on saved PNG/DOE (e.g., frequency-band inter-z and own-plane separation on a physically aligned grid). Give equations, minimal test script pseudocode, expected direction (NOT fabricated numeric gains), and what result refutes the story. Do not repeat D4/carrier/HF random diffuser or assume analytic backprojections collapse to a DC point. Do not change official Pearson/SSIM definition. Finally propose at most one model improvement only if scientifically justified, not currently being trained on Sanya GPU.
```

## Сессия 2 — G2 (ML/обучение)

```
LIVE FACTS:

One phase-only DOE 128² 8-bit, 4 distinct grayscale **amplitude** targets 128², z=24.1/25.3/26.5/27.7mm, λ=532nm, input pitch10µm. Single-FFT official Fresnel, native output pitch changes with z. Metrics Pearson and **global** SSIM on reloaded uint8 recon PNG. Inference = one trained neural forward; per-input iterative Adam/GS disallowed. Architecture 12ch = 4 targets + 4 complex inverse-Fresnel Re/Im (8 channels) feeding trainable STEM U-Net, original R3 objective.

Current strongest proven native-only model is v2 seed0 step12000 P=.61243 SSIM=.55356 A_P=.7733 B_P=.4516, B own 25/32. Asset 721714285, independently checked on Mac. B median has serious ghosting despite own 4/4. Earlier step900 P=.5778 S=.5067. B75 oversampling 25/75 neutral; WhiteCollars training harms our domain and rights not cleared; D4 c64 exact but model effect neutral; R15 +0.05 axial-loss neutral; R13 quadratic carrier NO-GO; random HF-basis not helpful. Sanya currently running wider/deeper network tests W32d3 W64d3 W32d4, 2 seeds, 12k steps; follow-on Adam-teacher distillation contemplated only if capacity not beneficial, under train-only constraints. No new Gemini task should duplicate GPU width jobs.

Adam 2000-step *per sample* from STEM initialization can reach P=.7952, B_P=.6526 on diagnostics, **NOT competition inference**. It suggests learnable gap; do not treat as neural model performance or guarantee. Geometric axial crosstalk is physically complex: do not assert four planes are fundamentally inseparable.

TASK:
You are an ML architect for amortized inverse problems with a single-neural-forward test-time budget. We have a strong but underfitting 12ch STEM, P=.6124, B=.4516, while per-sample Adam-on-phase teacher (diagnostic only, banned at inference) reaches about P=.795/B=.653. Sanya is *currently running width/depth W32d3/W64d3/W32d4*; do not duplicate that research. Evaluate at most THREE distinct train-time-only ideas, especially (a) teacher/student distillation from phase targets with circular phase ambiguity and multi-modal solutions; (b) teacher-guided image-domain optical losses using differentiable forward physics; (c) training B recombined independent quartets with valid split provenance. Include crucial pitfalls: direct angle regression across phase wraps, multiple equally good phases, teacher quality/latency, cost of generating 512 teachers, label leakage, confusion between phase teacher as training data and forbidden per-sample Adam at inference. Rank by expected improvement in B visual identity, coding time, training wall p50/p90, GPU/VRAM use, AI quota/reset risk, and ease of independent acceptance. Propose exactly ONE small controlled pilot suitable for the remaining hours. Provide formulas/pseudocode, failure tests for black/ghosted/noisy/swapped planes, 4x4 own-target matrix, B median/hard contact sheets, GO/NO-GO, and rollback. No invented test results or invalid licensing.
```
