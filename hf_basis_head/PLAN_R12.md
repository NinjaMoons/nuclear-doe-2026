# R12 STEM + HF-basis — pre-registered plan (written before any training run), 2026-11-07 ~20:20 MSK

Host: Mac M1 (8 cores), CPU, torch 2.9.0, python 3.11 venv; one training process at a time; Matvey's host untouched.
Code: branch research/r12-stem-hf-basis on top of 44f03784 (STEM loader); new files only in hf_basis_head/.
Data: manifest_expanded.json sha256 76d2474d...1c78 (corpus 6c6f136b...), TRAIN 512 / VAL 16. The stock driver
validates and loads every split (incl. TEST ids for disjointness) but VAL is the only split evaluated; no TEST metric.
Common (all arms, identical): driver tm.main via hf_basis_head.train_wrapper_hf, objective R3 =
research_r3_lsq_plus_0.3_centered_corr_plus_0.1_contract_v0 (bw.research_objective, == 9e8cd4a9 formula),
Adam lr 1e-3, batch 4, fp32, cpu, --threads 4, eval-every 100, seed-local batch generator (same batches for all arms),
same body/stem init for the same seed (tested), weights-only checkpoints every 100 updates.
Arms: A = stem (BackProjectionStemUNet, original scalar head);
      B = stem_const (residual head, 16 coefficient maps 32x32 -> bilinear, R_j = 1);
      C = stem_hf (same head, R_j = uniform[-pi,pi) seed 2026). B and C: +272 parameters each, zero-init.

Gate 1 (health): B and C, 100 updates, seed 0: finite objective/grad at every step, VAL eval runs, phase shape ok.
Gate 2 (pilot): A, B, C, 500 updates, seed 0, sequential.
Gate 3: if Gate 2 passes -> A, B, C 1000 updates seed 0, then seed 1 (time permitting). If Gate 2 fails -> NO-GO,
no full run.

Checkpoint per arm: (i) final step; (ii) selected by rule 6697684744 (max VAL mean Pearson; within 0.005 -> larger
worst-plane SSIM, then earlier step). Both reported; decision uses (ii) and must not contradict (i).
Metrics on VAL exports (official DOE.png uint8 -> reload -> 4 Fresnel -> 4 PNG -> reload): mean P/SSIM, A/B, per z,
worst plane; 4x4 C[k,j], own-argmax count, own margin; inter-z on z1 physical grid (full, low r<=0.15, band 0.15-0.5);
mask HF fraction (r>0.5 Nyquist of |FFT exp(i phi)|^2), R90, roughness; speckle: hot px (PNG>=0.9), top-pixel power,
bright fraction, mean level; train time; inference time. Paired per-quartet deltas; bootstrap over quartets (2000).

Gate 2 pass for C (all must hold, vs BOTH A and B at selected checkpoints):
  1. B-group own-argmax count of C > A and > B, OR B-group mean own-margin of C >= max(A,B) + 0.02;
  2. mean Pearson and SSIM of C >= max(A,B) - 0.005; A-group P and SSIM of C >= A-arm - 0.01;
  3. worst-plane P and SSIM of C >= A-arm - 0.01;
  4. no speckle degradation: mean hot-px fraction and top-pixel power of C <= 1.5x A-arm, mean PNG level >= 0.8x A-arm;
  5. visual sheet: no new visible noise floor relative to A.
Final GO (HF_BASIS_GO) needs Gate-2 criteria on 1000 updates for both seeds; one seed only -> INCONCLUSIVE (preliminary);
C only raising HF power without 1-3 -> HF_BASIS_NO_GO.
