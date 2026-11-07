# R13 Stage 0 — field diagnostic, no training (pre-registered 2026-11-07 ~17:58 UTC)
Issue #46. Code base: research/5-backprojection-stem @1790a37b (exact inverse_fresnel, official propagate_fresnel).
Quartets (fixed before measurement): B208:006 + first 8 A and first 8 B TRAIN keys in manifest 76d2474d order
(A64:000,001,002,003,004,005,007,009; B64:000,001,004,006,007,008,009,010) = 17 TRAIN quartets. No VAL/TEST.
Fields per z_k: B0_k = F^-1(T_k + 0j), Bc_k = F^-1(T_k * exp(i*psi)), psi = pi*(p^2+q^2)/128 on the centred output
pixel grid p,q = n - 64; T = PNG/255 amplitude. Check F(B) = T*exp(i*phase) numerically (rel L2).
Aperture metrics on |B|^2 (DOE plane, centred pixel radius r): energy fraction r<32, 32<=r<64, r>=64; radial profile.
Phase-only comparators (classical diagnostic, NOT a contest candidate): phi = angle(sum_k B_k) with |sum|<1e-12*max -> 0;
DOE.png codec save/reload -> official 4 Fresnel -> amplitude_to_u8 -> evaluate_multiplane vs targets:
P/SSIM per plane, own-target 4x4 C and argmax count, inter-z on physical grid, display brightness and speckle contrast.
Reference arm: STEM seed0 step900 checkpoint 569e1700 (one inference forward on these TRAIN quartets, no training).
Synthetic counterexamples: centred disk, off-centre square, uniform field, random binary speckle.
Decision: R13_STAGE0_DONE (benefit) only if phi_carrier vs phi_zero gives mean dP >= +0.02 AND B own-argmax not lower
AND mean dSSIM >= -0.01 over the 17 quartets; otherwise R13_STAGE0_NO_GO (Stage 1 not started). No training here.
