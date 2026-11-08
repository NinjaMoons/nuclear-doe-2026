# NIH-DIAG-SCIENCE-012 — why the one-pass STEM does not approach the Adam oracle · ggmaster913 · 08.11.2026 ≈03:45 МСК
READ_ONLY. No GPU, no TEST, no new training, no GO. Sources:
- main `fb1f855`;
- this branch `research/r19-xplane-contrast` (R20 `2619781`, R19 acceptance `01587a6`, R21 `8c270c4`);
- R19 `a07b6e9`;
- GRADIENT-014 `qa/matvey-diag-gradient-014@08a83f7`;
- #49 G0–G4 and OPUS_FINDINGS_HF_MASK (`research/49-gemini-intake`).

**The gap:**
- V2 VAL16 P .612 / B .452.
- Per-quartet Adam oracle with the **same R3** from the V2 phase: TRAIN-B P .66–.75 after 300–1000 steps; .795 / B .653 on diagnostics.
- Both are measured through the same official DOE.png → |U| PNG export.

## 1. R3 vs official metrics (code: `back_projection_stem/r3_loss.py`, `doe_models/train_multiplane.py:46-71`)

| R3 term (share of value, GRADIENT-014) | official counterpart | mismatch |
|---|---|---|
| LSQ-gain NMSE `Σ(g·A−T)²/ΣT²` with the optimal gain g (68–74 %) | Pearson on A/max | gain-invariant like Pearson, but not mean-invariant: it penalises DC/background offset that Pearson ignores |
| 0.3·mean(1 − corr(A/max, T)) (25–31 %) | Pearson | exact continuous surrogate; u8 vs continuous agree within 1e-4 (014) |
| 0.1·contract MSE (A/max − T)² (≈1 %) | global SSIM luminance/contrast on A/max | the only max-normalised term, so SSIM's mean/variance terms are trained with ~1 % weight |

- Missing from R3: no SSIM term, no u8 rounding, no DOE 8-bit decode.
- 014: cos(∇R3, ∇(1−P)) = .98 / .86, and the R3 direction gives 85–97 % of the Pearson-direction gain.

**Verdict:**
- **REJECTED** as the cause of the P gap: the oracle uses the same R3 and gets +.05…+.20 P.
- **HYPOTHESIS (minor, SSIM only):** SSIM is under-weighted, because only the 1 % contract term acts on A/max mean/contrast. This explains part of the SSIM gap, not P or ghosting.

## 2. The three hypotheses

| hypothesis | evidence | status |
|---|---|---|
| **H1 quantization / representation loss** (8-bit DOE, 8-bit PNG) | the oracle passes the same export and keeps its advantage; the phase step 2π/255 = .025 rad is tiny next to the oracle phase roughness (|∇θ| 1.1–2.5 rad/px); continuous corr = u8 P within 1e-4 | **REJECTED** as the gap cause (EXPORT-013 quantifies the residual) |
| **H2 cross-plane gradient conflict / optimization** | 014: off-diagonal cos −.21/−.23, cancellation ‖Σg‖/Σ‖g‖ .26–.31, yet −∇R3 still improves **every** plane. Adam with the same conflict reaches the oracle. R20: an explicit separation term only moves along the P/SSIM ↔ separation front (SSIM ↓ at every dose, 8/8). R15 axial ≈ 0 | **REJECTED as the root cause of the oracle gap.** CONFIRMED as a real property of the problem: it sets the trade-off front, which per-sample optimisation crosses and the amortised net does not |
| **H3 amortised inverse: expressivity / phase ambiguity** | (a) the oracle's gain lives in **coherent mask HF** (r > .5–.7 Nyq): low-pass r ≤ .7 removes it, mask HF .6–.8 vs V2 .15, R19 .17. (b) R21: equally good teachers (ΔP ≤ .009) have **unrelated** target-plane phases (agreement .01–.04 = chance), so the solution set is multi-modal and the per-target optimum is not a function of the targets. (c) Width/depth W32/W64/d4 neutral; R8-SCRATCH +.008, R18 +.002 (2 seeds), R19 +.005 (2 seeds): parameter count and heads move ≤ .008. (d) Random / incoherent HF fails (R12 head suppressed to .017 rad; diffuser). (e) BP-mix ceiling .593 vs equal-capacity .533 vs free Adam .667 | **CONFIRMED (where):** the gap is amortising **target-specific coherent HF phase.** **HYPOTHESIS (why):** the CNN's smooth/spectral bias plus averaging over a multi-modal solution set (R3 regression averages incompatible rough modes into a smooth phase), versus pure optimisation difficulty from the smooth V2 basin — separated by test C3 below |

## 3. Measurable criteria for the independent tasks (they run their own tests)

**EXPORT-013** (Sasha laptop, V2, 2 A + 2 B TRAIN):
- QUANTIZATION_ROOT_CAUSE=YES only if |B−A| + |C−B| ≥ .02 P on group B, i.e. ≥ ~10 % of the V2 → oracle gap. Prediction: ≤ .003.
- If time allows, the same A/B/C on one oracle phase. Rough phase could be more quantization-sensitive, so if the oracle loses ≥ .02 while V2 loses ≤ .003, the "same export" argument above needs revision.
- NO if < .005; INCONCLUSIVE in between.

**GRADIENT-014** (done, SUPPORTS partial):
- To make it causal, repeat in **weight space** (∂L_k/∂θ through the V2 Jacobian).
- Network-level conflict is CONFIRMED if the θ-space cancellation is markedly stronger than in φ-space (ratio ≤ .15 vs .26–.31). Under the same criterion, per-plane cos(g_k, ∇R3) ≤ 0 for any plane is also CONFIRMED.
- If θ-space ≈ φ-space, the conflict is the problem's own and not the network's: H2 stays rejected as the gap cause.

**R23 / CAPACITY-015** (Matvey HOME, pre-check only). Decisive test:
- **C3:** overfit a fresh STEM (R8-SCRATCH config) on **one** TRAIN-B quartet and on 64 TRAIN-B, same steps and seed; compare with the oracle on that quartet.
  - Expressivity/spectral bias is supported (Fourier/Gabor head justified) if the single-quartet net stays ≤ oracle − .05 P **and** mask HF ≤ .3.
  - Amortisation/ambiguity is supported (a new head would not help) if the single quartet reaches ≥ oracle − .02 while the 64-quartet net plateaus.
- **C4 "target-conditioned coherent HF, not texture":** swap the HF band (r > .5 Nyq) of the predicted phase between two quartets.
  - Coherent if the own-quartet P drops ≥ .02. Otherwise the HF is texture, i.e. the same failure mode as R12.
- Always report the matched R8-SCRATCH control, TRAIN-only (no VAL claims), official u8 P, B inter-z, and p90.

**R22** (GO s0 in #51/6714967745, both stages trainable): interpret as **feedback + capacity**.
- A gain ≥ .010 with lower B inter-z would be the first result crossing the front. It would support H3-b, i.e. physical feedback as a way to pick a mode.
- A gain ≈ .005 at the R19 level is consistent with the "smooth generic correction" failure mode.

## 4. Decision and next step
- **ROOT_CAUSE = H3** (amortised coherent-HF phase / multi-modal inverse).
- **H1 rejected; H2 is a property of the problem, not the gap's cause; R3 mismatch minor (SSIM only).**
- **Scientific next step:** no automatic training GO. Wait for C3/C4 (R23 pre-check) and R22 s0.
  - Only if C3 points to expressivity: R23 head GO (separate gate).
  - If C3 points to ambiguity: no architecture tweak is credible before freeze. V2 remains, and this diagnosis becomes the defense narrative.
