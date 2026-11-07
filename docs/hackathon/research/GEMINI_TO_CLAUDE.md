# Gemini → Grigoriy's Claude Code → experimental owners (no intermediary paraphrase)

## Workflow and truthful automation status

Two Gemini sessions can research in parallel, **but are not automatically connected to this GitHub repository merely because their prompts exist here**. They must submit the original answer in full, via copy/paste, saved file, or an already configured API workflow. `@ggmaster913` (Grigoriy's Claude Code) is the SCIENTIFIC GATEKEEPER: ingest raw responses unchanged and preserve the exact prompt+model+date, fact-check against actual optics/runtime and completed experiments, then publish a short CODE-READY hypothesis to an issue and assign the most suitable available owner. Do **not** route through Pavel/ChatGPT prose rewrites. Do not automatically start a GPU run from a suggestion.

## Live facts Gemini MUST receive

One phase-only DOE 128² 8-bit, 4 distinct grayscale **amplitude** targets 128², z=24.1/25.3/26.5/27.7mm, λ=532nm, input pitch10µm. Single-FFT official Fresnel, native output pitch changes with z. Metrics Pearson and **global** SSIM on reloaded uint8 recon PNG. Inference = one trained neural forward; per-input iterative Adam/GS disallowed. Architecture 12ch = 4 targets + 4 complex inverse-Fresnel Re/Im (8 channels) feeding trainable STEM U-Net, original R3 objective.

Current strongest proven native-only model is v2 seed0 step12000 P=.61243 SSIM=.55356 A_P=.7733 B_P=.4516, B own 25/32. Asset 721714285, independently checked on Mac. B median has serious ghosting despite own 4/4. Earlier step900 P=.5778 S=.5067. B75 oversampling 25/75 neutral; WhiteCollars training harms our domain and rights not cleared; D4 c64 exact but model effect neutral; R15 +0.05 axial-loss neutral; R13 quadratic carrier NO-GO; random HF-basis not helpful. Sanya currently running wider/deeper network tests W32d3 W64d3 W32d4, 2 seeds, 12k steps; follow-on Adam-teacher distillation contemplated only if capacity not beneficial, under train-only constraints. No new Gemini task should duplicate GPU width jobs.

Adam 2000-step *per sample* from STEM initialization can reach P=.7952, B_P=.6526 on diagnostics, **NOT competition inference**. It suggests learnable gap; do not treat as neural model performance or guarantee. Geometric axial crosstalk is physically complex: do not assert four planes are fundamentally inseparable.

## GEMINI 1 — OPTICS / SPECTRAL DIAGNOSTICS (read-only-first)

Prompt (paste all above + below):

> You are a skeptical Fourier optics physicist. Explain and quantify observed axial ghosting for our exact native-grid single-FFT Fresnel at four z. Distinguish mathematically limited low-frequency selectivity from recoverable model error (Adam teacher). Consider physical output scaling, Fresnel transfer, phase-only constraint, and finite aperture. Give a scientifically defensible 2-paragraph explanation and a slide-ready diagram proposal, explicitly mark approximations. Suggest no more than TWO unique, falsifiable read-only tests on saved PNG/DOE (e.g., frequency-band inter-z and own-plane separation on a physically aligned grid). Give equations, minimal test script pseudocode, expected direction (NOT fabricated numeric gains), and what result refutes the story. Do not repeat D4/carrier/HF random diffuser or assume analytic backprojections collapse to a DC point. Do not change official Pearson/SSIM definition. Finally propose at most one model improvement only if scientifically justified, not currently being trained on Sanya GPU.

Deliver `G1_optics_raw.md` and `G1_slide_claims.md`.

## GEMINI 2 — ML TRAINING / TEACHER-STUDENT ANALYSIS (costed and differentiated)

Prompt (paste all above + below):

> You are an ML architect for amortized inverse problems with a single-neural-forward test-time budget. We have a strong but underfitting 12ch STEM, P=.6124, B=.4516, while per-sample Adam-on-phase teacher (diagnostic only, banned at inference) reaches about P=.795/B=.653. Sanya is *currently running width/depth W32d3/W64d3/W32d4*; do not duplicate that research. Evaluate at most THREE distinct train-time-only ideas, especially (a) teacher/student distillation from phase targets with circular phase ambiguity and multi-modal solutions; (b) teacher-guided image-domain optical losses using differentiable forward physics; (c) training B recombined independent quartets with valid split provenance. Include crucial pitfalls: direct angle regression across phase wraps, multiple equally good phases, teacher quality/latency, cost of generating 512 teachers, label leakage, confusion between phase teacher as training data and forbidden per-sample Adam at inference. Rank by expected improvement in B visual identity, coding time, training wall p50/p90, GPU/VRAM use, AI quota/reset risk, and ease of independent acceptance. Propose exactly ONE small controlled pilot suitable for the remaining hours. Provide formulas/pseudocode, failure tests for black/ghosted/noisy/swapped planes, 4x4 own-target matrix, B median/hard contact sheets, GO/NO-GO, and rollback. No invented test results or invalid licensing.

Deliver `G2_ml_raw.md`, `G2_pilot_brief.md`.

## Grigoriy Claude intake gate

1. Read originals, save full raw prompts/answers and provenance to **git branch/worktree**, e.g. `docs/hackathon/research/gemini/G1_optics_raw.md`, `G2_ml_raw.md`; never replace original answer with assistant summary or lose references.
2. Cross-check factual claims with `doe_core/fresnel.py`, `back_projection_stem`, `r3_loss.py`, Github #39/#43/#44/#45/#46/#47/#48/#40/#41, actual numerical evidence. Distinguish `CONFIRMED / MODEL_APPROXIMATION / HYPOTHESIS / REJECTED`.
3. Create `IDEA_CARD.md` or issue comment per credible hypothesis: explicit target, exact diff to baseline, train-only/test-time eligibility, owner + host, dependencies, p50/p90 measured or UNKNOWN, quota/reset known or UNKNOWN, prereg 100-step health, ≤300–500 matched pilot, checks, expected artifacts. Include whether this duplicates Sanja current width/depth or Matvey existing work.
4. Scientist owns independent checks; GPU experiment assigned only if owner/host confirmed resource available. Coordinator #8 must receive a link **directly to the GitHub artifact**, not a paraphrased idea.
5. After experiment DONE/NO_GO, keep entire negative result in the research record. Immediately select next high-value task or #47 defense explanation.
6. Research STOP gates: no new training after 02:00 MSK; all research evaluation by 04:00; model+deck evidence freeze 05:30. Prefer defense deliverables if p90 overruns.
7. More important than speculative new physics: preserve a truthful 4-plane visual record of what the selected neural model actually reconstructs.

**If both Gemini sessions exhaust limits:** Grigoriy's Claude already has a defensible evidence base and should transition to `docs/hackathon/defense/DEFENSE_INDEX.md` and #47, not wait for further prompts.
