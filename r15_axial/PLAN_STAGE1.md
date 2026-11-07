# R15 Stage 1 — matched fine-tune, pre-registered before running (2026-11-07 ~18:25 UTC), issue #48
Both arms start from the frozen STEM checkpoint 569e170065cca160eb6ce64a0707d81e5e016857b0962498c156cba80f6d3b7d
(identical weights), fresh Adam (checkpoint has no optimizer state -> parity), same seed 0 -> same batch sequence,
lr 1e-3, batch 4, fp32, CPU 8 threads, TRAIN512 / VAL16 manifest 76d2474d, eval-every 100, 300 updates.
- base: R3 (research_r3_lsq_plus_0.3_centered_corr_plus_0.1_contract_v0)
- treatment: R3 + 0.05 * L_ax (research_r15_r3_plus_0.05_axial_innovation_v0), lambda locked
Health gate = step 100 inside the run: finite objective/grad at every step, VAL export at step 100 works; else STOP.
Evaluation: official DOE.png uint8 -> reload -> 4 Fresnel -> 4 PNG -> reload (in-process stock evaluate_checkpoints).
Readouts: step 300 (primary) and rule 6697684744 per arm; P/SSIM A/B, per z, worst plane, B own-argmax + diag-offdiag,
inter-z physical, brightness, speckle; paired bootstrap over quartets (n=16). Also frozen STEM step900 (=step 0) as ref.
Visual (predeclared VAL samples): A median A64:010, B worst B64:002, B median B64:012, B strongest B64:032;
rows TARGETS / frozen STEM / base 300 / treatment 300, raw official PNG, no post-processing.
Decision: PILOT_VISUAL_GO only if dP >= +0.01, dSSIM >= -0.005, B own-argmax not lower AND the visual sheet shows
fewer ghosts on the predeclared B samples without lost shape/contrast or more speckle; NO_GO if numbers worse
(dP <= -0.005 or dSSIM <= -0.01) or visually worse; otherwise INCONCLUSIVE. No TEST, no package change.
