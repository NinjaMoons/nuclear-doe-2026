# R12 STEM + HF-basis — report (research only), 2026-11-07

Verdict: **HF_BASIS_NO_GO** for this parameterisation at the pre-registered pilot budget (Gate 2 failed; no full run).

Host: Mac M1, CPU, torch 2.9.0, 4 threads; one process at a time. Branch research/r12-stem-hf-basis (on 44f03784).
Arm A (STEM) on this Mac reproduces Matvey's Windows STEM seed0 run: VAL P/SSIM at steps 100/300/500 equal to 4 decimals
(0.5203/0.4274, 0.5517/0.4579 vs 0.5516/0.4577, 0.5691/0.4949), identical first objectives.

Tests: hf_basis_head/test_hf.py 13/13 OK (unittest_log.txt). Gate 1: B and C 100 updates, finite, VAL eval ok.

Gate 2 (500 updates, seed 0, same batches/init, R3 objective, lr 1e-3, batch 4), VAL n=16, rule checkpoint = step 500 in all arms:
| arm | P | SSIM | A P/SSIM | B P/SSIM | B own-argmax | B margin | B inter-z full/low/band | worst plane P/SSIM | HF | hot px | top px |
| A stem | .5691 | .4949 | .7220/.6683 | .4163/.3216 | 24/32 | .090 | .681/.941/.621 | .547/.446 | .106 | .0005 | .0009 |
| B const | .5665 | .4917 | .7191/.6615 | .4140/.3220 | 24/32 | .090 | .687/.942/.630 | .542/.449 | .108 | .0004 | .0009 |
| C hf | .5680 | .4941 | .7206/.6676 | .4154/.3207 | 24/32 | .088 | .680/.943/.618 | .546/.449 | .107 | .0005 | .0009 |
Paired per quartet (bootstrap 2000 over quartets, mean [95% CI]): C-A dP -0.0011 [-0.0016,-0.0007] (0/16 better),
dSSIM -0.0008 [-0.0032,+0.0015]; dB_margin -0.0026 [-0.0040,-0.0009]; C-B dP +0.0015 (13/16), dSSIM +0.0024 (10/16).
Gate-2 criterion 1 fails (own-argmax 24 = 24, margin not +0.02); C is not better than A.

Mechanism: learned residual on VAL inputs — C (HF): dphi RMS 0.019 rad at step 100 -> 0.017 rad at step 500, coefficient
RMS 0.0029 -> 0.0026 (shrinking); B (const): dphi RMS 0.24 -> 0.32 rad. The shared network actively keeps the random-HF
term near zero under R3; HF power of C equals A (0.107 vs 0.106). This is not a budget limit: Adam could move the
zero-init weights by O(lr*steps)=0.5, they stayed < 0.01.

Limitations: one seed, 500 updates; one basis family (white uniform[-pi,pi), seed 2026) and one coefficient scale;
per-quartet optimisation results (Gemini simulation, #14 teacher) are not contradicted — they optimise a_j per input,
which a shared network does not learn here. Not tested: other basis spectra/scales, larger lr for the head, 1000 steps.

Reproduce (cwd = this worktree, venv with torch 2.9.0):
  python -m unittest hf_basis_head.test_hf -v
  for arm in stem stem_const stem_hf: python -m hf_basis_head.train_wrapper_hf --arm $arm --manifest <corpus>/manifest_expanded.json
      --out-dir <runs>/pilot500_s0_$arm --device cpu --updates 500 --eval-every 100 --threads 4 --seed 0
  python -I hf_basis_head/analyze_r12.py <runs>/pilot500_s0_analysis.json A=<runs>/pilot500_s0_stem B=<runs>/pilot500_s0_stem_const C=<runs>/pilot500_s0_stem_hf
Time: tests 1.2 s; Gate 1 2 x 45 s; Gate 2 train 128.5/139.6/132.7 s + VAL export ~2 min per arm; 0.23 s/update.
