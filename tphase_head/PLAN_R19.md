# R19 · learned target-plane phase on the STEM — pre-registered 08.11.2026 ~02:15 MSK (before any full run)

Author: Opus (ggmaster913, #49 science gate). Directive: #49/coordinator-r19-research-request, #51 superseding 01:54.

## Hypothesis (one factor)
The B deficit is amortisation of target-specific coherent HF DOE phase (#49: F1 teacher gain lives in HF > .7 Nyquist;
G4 M0–M4: HF must be coherent with its own LF, an HF patch on V2 LF gives +.012 vs oracle +.23; random HF of the same
power gives −.14). In V2, R8-RES and R18 the physical features are B_k = P_k⁻¹(T_k·e^{i·0}) with **zero target-plane
phase**. The variable that per-quartet Adam/GS actually shapes is the target-plane phase (teacher roughness R_ψ ≈ 3.1 rad
vs V2 0.07–0.6). R19 lets the network **predict θ_k from the four targets**. Then B'_k = P_k⁻¹(T_k e^{iθ_k}) gives
coherent, target-conditioned DOE-plane HF analytically, through the exact inverse Fresnel, not through DOE-plane convs.
- Distinct from R8/R18/STACK: those re-weight fixed zero-phase B_k in the DOE plane. R19 changes *which* back-projection is formed.
- Distinct from loss variants (R15 / contrastive): R19 does not change the objective.

## Model (`tphase_head/model.py`)
- θ = π·tanh(CNN(T)), with CNN = Conv5×5(4→32)–GELU–Conv5×5(32→32)–GELU–Conv3×3(32→4); last layer zero-init.
- B'_k replaces B_k in the 12 STEM channels.
- Direct analytic path: phase = arg(e^{iφ_STEM} + Σ_k g_k·B'_k), with g_k complex scalars, zero-init.
- At init the output equals V2 exactly (test, < 1e-5 rad).
- Rules: one forward pass; only the analytic inverse Fresnel of the targets, as V2 already does. No forward Fresnel inside the network, no iteration, no optimizer at inference.

## Arms (matched, seed 0 pilot)
Both arms use the same settings:
- init v2 19b05dad; fresh Adam lr 1e-3; batch 4;
- sampler `randperm(512, Generator(30000+seed))[:4]`; native TRAIN512 only (manifest 76d2474d);
- R3 (`bw.research_objective`); 6000 updates; fp32 (θ path complex128);
- evaluation at step 6000 only, VAL16 official export (DOE.png u8 → reload → 4 Fresnel → 4 PNG → reload).

The arms:
- `control` — STEM fine-tune, byte-identical code path to the R18 control;
- `tphase` — the R19 model.

No TEST. No new data. No VAL tuning: one pre-registered configuration, no hyper-parameter search.

## Gate (seed 0, tphase − control on the same host)
- **SIGNAL → replicate seed 1**, if all hold:
  - ΔP_all ≥ +.010 **and** ΔB_P ≥ +.010;
  - B own-target ≥ control; worst-plane P ≥ control − .005; SSIM ≥ control − .005;
  - B inter-z (full, z1 grid) ≤ control;
  - lit fraction, hot and top-pixel power each within ±25% of control (no dimming/brightness/speckle collapse).
- **INCONCLUSIVE:** +.003 ≤ ΔP < +.010 → report, no replication unless time remains before 04:30.
- **NO_GO:** otherwise, or if the mechanism is unused (mean |θ| < .05 rad or Σ|g_k| < .01 on VAL16), or if the run is non-finite.
- Promotion is not part of this pilot. V2 asset 721714285 stays the submission unless a separate peer-accepted gate is met.

## Readouts
P, SSIM, A/B P and SSIM, B own/32, B margin, worst plane P/S, B inter-z full/low/band, mask HF, lit/hot/top,
θ roughness (mean |∇θ|) and |g_k|, paired bootstrap over the 16 quartets, sheets for A64:060, B64:012, B64:016.
Analysis: `hf_basis_head/analyze_r12.py` (r12 branch), with the same JSON shims as R18.

## Cost
- Mac M1 CPU 2 threads: smoke of 20 updates plus VAL eval passed (P .608, finite).
- RTX 4070 Ti estimate: ≤ 2× STEM step → 6000 updates ≈ 4–6 min per arm plus eval ≈ 2 min. p90 ≤ 15 min per arm, ≤ 30 min both.
- Mac fallback: the tphase arm only, CPU, vs the existing R18 s0 control (the same code/seed/sampler/host, ckpt e54f52f8). ≈ 40–60 min.

## Tests
`R19_V2=<v2.pt> python -m unittest tphase_head.test_r19` → 2/2 OK:
- init == V2 and new parameters get gradients;
- θ changes the output, and the checkpoint round-trips.
