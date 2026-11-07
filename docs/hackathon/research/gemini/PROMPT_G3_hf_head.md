# G3 prompt (one web-Gemini call) — prepared by @ggmaster913, 08.11 ~00:46 MSK. Paste EVERYTHING below the line.
---
You are a skeptical computational-optics / inverse-problems reviewer. Check a specific hypothesis and our math; do NOT
give a general survey. No invented numbers: when you estimate, say "estimate" and how.

FACTS (measured by us, official single-FFT Fresnel, N=128, pitch 10 um, lambda 532 nm, z = 24.1/25.3/26.5/27.7 mm,
phase-only DOE quantised to uint8 phi = -pi + 2*pi*c/255, targets = amplitude PNG/255, metrics on uint8 |U|/max PNG:
Pearson + global SSIM). Contest rule: ONE neural forward at inference, no per-input optimisation loops.
- Network: 12-ch STEM U-Net (inputs T_k and Re/Im of B_k = P_{z_k}^{-1}(T_k + 0j) normalised by RMS; P^{-1} is the exact
  discrete inverse of the single-FFT Fresnel), output phi via 1x1 conv; R3 loss = LSQ-gain NMSE + 0.3(1-corr) + 0.1 MSE.
  VAL16: P .612 (Group B with independent planes .452). Width/depth, phase-space distillation from Adam teachers,
  B oversampling, D4 augmentation, axial-innovation loss, a fixed random HF basis head, B recombination: all neutral.
- F1: Adam per-quartet teachers lose most of their gain when exp(i phi) is low-passed to r <= 0.7 Nyquist
  (B208:006 P .661 -> .399). Network masks keep 87-97% of power below r 0.7 and are insensitive to that low-pass.
- F2: Adam warm-started from the network phase raises P .461 -> .667 while mask HF fraction (r>0.5 Nyq) rises .21 -> .69;
  after low-pass r<=0.7 the gain disappears (P .449). Inter-plane correlation of reconstructions does not drop.
- F3: per-quartet optimisation of phi = arg((1+a0) e^{i phi_net} + sum_k a_k B_k) with SMOOTH complex a (32x32, bilinear
  to 128) reaches P .593 / .512 (two TRAIN-B quartets) vs .461/.427 network, .533/.483 for an equal-capacity smooth
  control without B_k, .667/.572 free-phase Adam.
- Running now: (R18) network head producing a_k maps from decoder features (zero-init, exact network at step 0);
  (R8-RES, teammate) phi = phi_net + atan2(im, re), (re, im) = 1x1 conv over decoder features + the 8 Re/Im B_k channels,
  init bias (1,0).
- RESULT R8-RES (fine-tune from the network, 6000 steps, 2 seeds): dP +.0002 [-.0023,+.0020]; the head is used
  (|w| ~.85, mean |atan2| .16 rad) but (re,im) stays far from zero, mask roughness stays .08 -> gradient descent on R3
  from the smooth network basin does not exploit the HF path. A from-scratch variant (random phasor init) is running.

QUESTIONS (answer each, with equations):
1. Is F2 consistent with physics? Explain why per-quartet gains need mask structure at 1-3 pixel scales for our
   geometry (N dx^2 / lambda = 24.06 mm ~ z_1), or show where our interpretation is wrong (e.g. aliasing, quantisation,
   speckle that inflates Pearson). Give one cheap read-only check that would refute F2.
2. Math check of arg((1+a0) e^{i phi} + sum a_k B_k): failure modes for gradient training (zeros of the sum -> phase
   singularities/vortices, gradient blow-up, scale ambiguity between a_k and B_k RMS normalisation). Propose a minimal
   safeguard that keeps step-0 equality with the base network.
3. Spatially varying smooth weights (R18) vs global 1x1 mixing (R8-RES): which should learn better across quartets and
   why (in terms of what varies between B_k at different z and across the aperture)? Give a falsifiable prediction.
4. Give at most TWO alternative single-forward ways to produce target-specific HF phase that are NOT in the list above
   (e.g. phase from a learned complex combination of B_k with non-zero target phase priors, multi-scale B_k, learned
   per-plane focusing kernels). For each: exact formula, why it differs from R18/R8-RES, cost, and a counterexample where
   it cannot help.
5. Given the R8-RES null: what single-forward training change would move the solution OUT of the smooth basin
   (e.g. curriculum, initialisation, objective term) without per-input optimisation at inference? One proposal, one
   prediction, one refutation condition.
Answer compactly; mark every claim as derived / estimated / speculative.
