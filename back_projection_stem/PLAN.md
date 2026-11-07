# Back-Projection Stem — PREP PLAN (written before any code/run), 2026-11-07 ~14:00 UTC

Tasks: #5/6707710203 (Pavel, READY/PREP ONLY) and #5/6707752950 (coordinator, ACTIVE prototype).
Base: research worktree at 49b96041 (TRAINING_SHA), local branch `research/5-backprojection-stem`, NOT pushed,
NOT for merge. Only `back_projection_stem/` is added. Official Fresnel, production U-Net, driver, CLI, best.pt: read-only.

## Scope (what is done)
1. Device check: torch.cuda.is_available(), GPU name/driver/torch CUDA. Machine has no NVIDIA GPU -> CPU tests; VRAM = PENDING.
2. Exact inverse of the official single-FFT Fresnel (doe_core/fresnel.py), reusing its _centered_axis/_chirp, carrier,
   scale sqrt(HW)*dy*dx and output pitch:  F(U) = o * fftshift(fft2(ifftshift(c_in*U), ortho)), o = s*c_out
   F^-1(V) = fftshift(ifft2(ifftshift(V/o), ortho)) / c_in.  No z -> -z, no sqrt(T), forward untouched.
3. Features: V_k = T_k + 0j on the native output grid of z_k (zero output phase = feature heuristic, NOT a physical
   reconstruction); B_k = F_k^-1(V_k) in complex128; each B_k divided by its own RMS (zero-RMS -> zeros).
   Channel order fixed: [T1,T2,T3,T4, Re1,Im1, Re2,Im2, Re3,Im3, Re4,Im4], shape [B,12,H,W], float dtype of targets.
   T channels use the production affine 2T-1 (same as FourInputUNet).
4. Model: BackProjectionStemUNet = FourInputUNet with encoders[0] = _block(12, base); every other block, the wrapped
   phase head [B,1,H,W] and the R3 loss unchanged. Input stays targets [B,4,1,H,W]. Separate hparams
   (in_channels 12, stem backprojection_v0) so the production loader can never confuse it with best.pt.

## Tests (unittest, stdlib; must all pass before any pilot)
- round trip both directions F^-1(F(U)) and F(F^-1(V)) on 4x6 and 5x7 grids with dy != dx, several z,
  complex64 (rel L2 < 1e-5) and complex128 (< 1e-12); 128x128 on all 4 official z, B=2.
- independent reference: direct-sum DFT forward from the continuous formula (centered indices, explicit exp) and
  numpy linalg.solve as inverse on 4x6 and 5x7; analytic inverse must match solve() and matrix product ~ I.
- dtype/device preserved; real input rejected.
- features: shape [B>1,12,H,W], channel order, Re/Im equal F^-1(T+0j)/rms, NOT F^-1(sqrt(T)) for non-binary T.
- zero / constant / random targets: finite features, finite R3 loss, finite gradients after backward.
- output phase [B,1,H,W] in [-pi,pi); production FourInputUNet still 4 channels and best.pt still loads strictly.
## Profiling
CPU fp32, 8 threads, batch 4, 128x128, real Fresnel + R3 loss (beta 0.3): time of one optimizer step
(forward+backward+Adam) for 4-ch vs 12-ch, plus feature-construction time; 3 warm-up + 10 timed steps.
## Not done
No training of any length, no TRAIN/VAL/TEST change, no GPU pilot (GPU PILOT AUTHORIZED? NO).
STOP on any failing test with the exact log.
