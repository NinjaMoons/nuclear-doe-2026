# NIH-GEMINI-SCIENCE-028: receipt and technical red-team

**STATUS: POST-SUBMISSION RESEARCH / NOT a submitted model / NOT an empirical improvement.**
**Author of original AI report:** Alexander's Gemini (provided by Pavel; AI-generated, not independently verified).
**Original uploaded filename:** `message (5)(1).txt`, 72,627 bytes, 678 logical lines.
**Original raw-file SHA256:** `719eca2c19e46ef7fedcb778ecc5acf37b6ede81ef3753d7b2afb68217e06a8b`.
**Original archived blob SHA (verified exact bytes):** `e407f39e9a791865d3425967339869833827c666`.
**Archived commit:** `649e0abfe0cc61ea205d3394d1c8df4f20d7afa8`.
**Original:** [GF_DDFN_GEMINI_RAW_2026-11-08.md](GF_DDFN_GEMINI_RAW_2026-11-08.md).
**Extracted unmodified Python blocks:** [gradient diagnostic](diagnose_grad_conflict_GEMINI_UNTESTED.py) and [model prototype](gf_ddfn_GEMINI_UNTESTED.py); Python AST parsing PASS, PyTorch/runtime/physics **NOT TESTED**.

## What is supported vs still speculative

- TRAIN quartets C3: one-example `P=.6562` after 3000 updates, N64 `P=.4938` after about 188 exposures per example; unequal training exposures are a real confound. Do **not** compare these TRAIN results causally to V2 `VAL16=.61243` because datasets differ.
- Multi-plane `z=[24.1,25.3,26.5,27.7] mm`, `N=128`, `dx=10um`, `lambda=532nm`; single-FFT output pitch depends on z. The quoted DOF=1.74mm uses **one convention** (`2 lambda/NA²`), not a rigorous barrier that no better DOE can overcome.
- `eta=||sum_i g_i||/sqrt(sum_i ||g_i||²)` and `eta<1` quantify net negative pairwise cross terms at a chosen checkpoint; correct diagnostic (with epsilon/zero-grad precautions). But **kernel-weight HP gradients are not equivalent to the high-frequency spectrum of the actual 128x128 DOE**; no direct causal proof of phase mode averaging from the proposed statistic.
- STEM V2 **already contains inverse-Fresnel backprojected Re/Im fields** as 8/12 input channels, so the proposed GF-DDFN is **not** the first physics domain-bridging implementation. Novel aspects are target-conditioned complex-field phases, fixed HF phase anchor and optional dual-expert routing; prior R19 used learned phase differently and R22 feedback differently.
- Claim that model learns a conditional mean of physical phases is NOT mathematically proved: baseline optimizes a differentiable amplitude/physics loss and is not supervised to match arbitrary oracle phase maps.
- The report's percentages for H1/H2/H3/H4, 85/90% confidence, an "oracle ceiling" from one example, claims of guaranteed HF gains, runtime 1.8ms, <2MB VRAM and "100% contest compliance" **are unmeasured/unsupported**. Literature gives background, not measured validation for this specific model.
- `gf_ddfn_GEMINI_UNTESTED.py`: exact adjoint must match production Fresnel FFT normalization, physical pixel-weighted inner products, z-dependent scaling/chirps/fftshift. Current code uses `ifft2(norm="ortho")` and omits the forward normalization prefactors; "exact adjoint" requires proof by complex inner-product test.
- `STEQuantize8Bit` currently uses `round(255*q)/256` while official u8 decoding requires `round(255*q)/255` with the exact production phase wrapping convention. Fix in an EXPERIMENTAL WORKING COPY only, never modify this archived original.
- `diagnose_grad_conflict.py` uses `loss_fn(pred,x_i)` and unconditional `p.grad.detach()`. Production R3 physics loss has to be adapted exactly, and unused parameters/zero norms handled, before running. Parameter keyword heuristic ("out") could misclassify the network head.
- Training/eval Gumbel soft router vs deterministic top-1 route may introduce a mismatch and branch-collapse; isolate learned target phases *without* router or amplitude precompensation first.
- Code AST syntax PASS only: no checkpoint trained, no measured P/SSIM/ghosting, no evidence improved versus baseline.

## Smallest empirical plan, no mutation of submission

1. **QA/diagnostic (CPU or GPU)**: fix test harness in a *separate* branch. Verify official R3 physical loss and train-only 4 A + 4 B fixed examples. Compute actual per-quartet parameter gradients and pairwise cosine, eta, A/B norm ratios; also compute output DOE phase spectrum separately, not inferred from conv kernel.
2. **Physics contract first**: candidate forward/adjoint complex inner-product equality against production 4-z Fresnel with its native nonuniform output physical grid and FFT conventions, gradients finite, u8 roundtrip exact.
3. **One-factor pilot**: fixed target-conditioned learned `delta_psi(T)` atop existing protected baseline physics, NO 2-expert router, NO amplitude precompensation, NO added loss/sampling/validation sweep; matched update/time budget and seed vs code-level control. Only if that works, add an individual router experiment.
4. **Official evidence**: TRAIN-only diagnostics, independent fixed endpoint VAL16; saved/reloaded DOE.png L8 + four reloaded amplitude PNGs, four Pearson/global SSIM, A/B, B own-vs-other crosstalk, worst plane, brightness/speckle, exact SHA and source version. No TEST.
5. **Promotion after deadline**: keep on post-submission research branch and explicitly label results as *after contest submission*, including when demonstrated in presentation. Never rewrite `main` or snapshot `submission/official-stem-v2-20261108`.

## Primary literature (background, not direct validation)
- Rahaman et al. (ICML 2019), spectral bias: https://proceedings.mlr.press/v97/rahaman19a.html
- Eybposh et al. (Optics Express 2020), DeepCGH: https://pubmed.ncbi.nlm.nih.gov/32906933/
- Peng et al. (ACM TOG 2020), Neural Holography: https://doi.org/10.1145/3414685.3417802
- Yu et al. (NeurIPS 2020), PCGrad: https://arxiv.org/abs/2001.06782
