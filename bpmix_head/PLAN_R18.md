# R18 STEM + complex back-projection mixing head — pre-registered 08.11.2026 ~00:33 MSK (before any training)

Evidence (read-only, #49, tests/*.py on research/49-gemini-intake): warm Adam teacher from v2 gains P only together with
mask HF (B208:006 P .461->.667, HF .21->.69); removing r>0.7 Nyquist of the mask removes the gain (P .449). Per-quartet
ceiling of phi = arg(c0*exp(i*phi_v2) + sum_k c_k*B_k) with smooth 32x32 complex c (B_k = STEM back-projection features):
P .593 / .512 vs equal-capacity smooth control without B_k .533 / .483 (B208:006 / B64:010).
Hypothesis: giving the network this physically structured HF path (target-specific, unlike R12 random basis) improves VAL.

Model: BackProjectionStemUNet body + coefficient head (1x1 conv, 10 channels = Re/Im of a0..a4) on the last decoder
features, adaptive-avg-pooled to 32x32, bilinear (align_corners=False) to 128; c0 = 1 + a0, c_k = a_k;
phi = angle(c0*exp(i*phi_stem) + sum_k c_k * (Re_k + i Im_k)), Re/Im = the existing STEM feature channels 4..11.
Head zero-init -> phi == STEM output at step 0 (up to angle/wrap representation). One forward at inference.
Arms (matched, seed 0, Mac M1 CPU, one process at a time): init v2 19b05dad (head zero for treatment), fresh Adam lr 1e-3,
batch 4, R3 objective, TRAIN512 (manifest 76d2474d), sampler randperm(512, Generator(30000+seed))[:4] for both arms,
6000 updates. control = v2 + R3 fine-tune (same everything, no head). Eval: step 6000 only, VAL16 official export.
GO (research candidate only): dP >= +.01 vs control, dP_B >= +.005, B own not lower, SSIM not lower, no speckle increase
(hot px / top px <= 1.5x); NO_GO if dP <= -.005; else INCONCLUSIVE. Seed 1 only if time allows before 02:00 start gate.
