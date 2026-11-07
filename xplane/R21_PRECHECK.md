# R21 pre-check (teacher target-plane phase consistency / roughness) → **SCIENCE_GATE = NO_GO** · ggmaster913 · 08.11 ≈02:58 МСК

**Question (coordinator, #49 6714584394).** In R19, θ_k stays smooth: |∇θ| .048 rad/px, while the teacher's is ≈ 2–3 rad/px. Would a train-only teacher-consistency term on θ_k, or a θ-roughness prior, close the gap?

**Pre-check.** `xplane/teacher_theta.py`, 4 TRAIN-B quartets, CPU, no GPU, no TEST.
- **Teachers:** Adam on R3, lr .05, 1000 steps, started three ways: V2 phase + N(0, .1 rad) with seeds 0 and 1, and a uniform random phase.
- **Measured:**
  - P, SSIM and own on the official PNG;
  - |∇θ_k| over lit target pixels;
  - circular agreement |⟨e^{i(θ_a−θ_b)}⟩| over lit pixels (1 = the same phase up to a constant, ≈ 0 = unrelated).

| quartet | P: V2 / teacher n0 / n1 / rand | |∇θ|: V2 / n0 / rand | agree n0~n1 | agree n0~rand | agree n0~V2 |
|---|---|---|---|---|---|
| B64:010 | .427 / .569 / .569 / .563 | .30 / 1.80 / 2.35 | .56 | **.03** | .48 |
| B208:006 | .461 / .665 / .665 / .661 | .79 / 2.10 / 2.40 | .61 | **.04** | .43 |
| B208:127 | .605 / .755 / .755 / .746 | .34 / 1.15 / 2.37 | .72 | **.01** | .70 |
| B208:046 | .519 / .698 / .698 / .696 | .46 / 2.08 / 2.47 | .59 | **.04** | .47 |

**Findings**
1. **Equally good teachers have unrelated target-plane phases.** The random-init teacher matches the warm teacher's P (ΔP ≤ .009), but their θ agree only at .01–.04, i.e. chance.
2. **θ_teacher is not a function of the targets.** The warm teacher's θ inherits from its initialisation (agreement with V2 .43–.70), and two nearby inits give .56–.72.
3. **A teacher-consistency loss on θ would therefore regress a multi-modal, init-dependent label.** That is the ill-posed "circular ambiguity / many equally good phases" pitfall (G2). The expected outcome is averaging toward a smooth θ, i.e. the R19 state.
4. **Roughness is a property shared by all good solutions, but a roughness prior does not say *which* rough phase.** The gain needs a θ that is coherent across the 4 planes and with the mask. Uncoherent HF has already failed: random diffuser, R12 HF basis suppressed to .017 rad, and the LPF test showed the gain lives in *coherent* HF.
5. **No train-only label or prior makes this coherence learnable in one forward pass.** The pre-check therefore shows no gap that such a term could close, so the gain of R21 > +.010 is not credible.

**Decision.**
- R21 (teacher-θ consistency / θ roughness, from scratch or continued) = **NO_GO**.
- The GPU stays free for P0 artifact checks. V2 remains the baseline.
- R19 science stays as is: replicated B-specific +.005 / +.009, below the gate.
