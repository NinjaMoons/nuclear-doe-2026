# G4 (Gemini one-shot HF critique) — verification by Opus, 08.11 ~02:00 MSK

Raw: `G4_oneshot_hf_raw.md` (model UNKNOWN). Test: `tests/g4_m04_test.py` → `tests/g4_m04_result.json`, no training,
official export (DOE.png u8 → reload → 4 Fresnel → 4 PNG). Masks: V2 one-forward vs Adam oracle (NOT contest) from
#43 evidence asset 722010850 (sha 98302e47…): A64_060, B64_012, B64_016 (VAL). n = 3 quartets, so this is diagnostic.

## Claims
| # | Gemini claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Sec.1 physics: Δψ = πλΔz f², walk-off λΔz f, separation needs high f | MODEL_APPROXIMATION | Qualitatively consistent with F1. The "≥0.7 Nyquist threshold" isn't derived; the numbers come from a paraxial plane-wave picture |
| 2 | Error 1: our LPF test broke \|U0\|=1 | REJECTED | `lpf_mask_test.py` already takes arg(LPF) → unit modulus is restored. Same result here: M1 (B .472/.399 vs Adam .681/.665) |
| 3 | Error 1: LPF of the U0 spectrum crops the target periphery (FoV) | REJECTED | The output is F{U0·Q}, not F{U0}. Measured: under LPF the Adam gain is lost **equally** in the centre (\|p\|,\|q\|<45) and on the periphery: B64_016 centre .637→.397, periphery .586→.341 |
| 4 | Error 2: the bpmix ceiling is "strawman"; GS-0 arg(ΣB) gives P≈.48–.54 | PARTLY REJECTED | GS-0 on B was measured: B64:010 .29/.04/.28/.57, B208 .27/.05/.17/.66 — not .48–.54. The equal-capacity control (smooth basis) was a fair one. The point that "per-quartet fitting ≠ amortization" is CONFIRMED by R18 (+.004 in the trained network) |
| 5 | "\|B_m\| is concentrated at DC, near zero at HF" | REJECTED (earlier) | Energy inside r≤16 px is 3–22% (#49, G0) |
| 6 | Q8 codec 2π/256·round(255/2π·…) | REJECTED | The contract is encode floor(255(φ+π)/2π+0.5), decode −π+2πc/255. The test uses the official codec |
| 7 | M2/M3 falsification test | RUN, see below | — |
| 8 | TargetPhaseWirtinger STEM: forward→inverse inside the network | HYPOTHESIS, NO_GO tonight | It is one GS step inside the forward pass (rules risk, see #49 G0 card). It also needs training, and the 02:00 gate has passed |
| 9 | Jury text: "spectral bias → smooth basin" | CONFIRMED in substance | R_ψ teacher ≈3 vs V2 .07–.6; but use our numbers, not Gemini's |

## M0–M4 result (mean Pearson of 4 planes, own-argmax)
| mask | A64_060 | B64_012 | B64_016 |
|---|---|---|---|
| M0 V2 one-forward | .830 4/4 | .469 4/4 | .409 2/4 |
| M1f Adam oracle (full) | .951 | .681 | .665 4/4 |
| M1 Adam LPF ≤0.7 (unit modulus) | .929 | .472 | .399 4/4 |
| M2 LF V2 + HF Adam (coherent) | .817 | .506 | .397 3/4 |
| M3 LF V2 + HF same power, random phase | .807 | .348 | .250 2/4 |
| M4 LF V2 + HF one-step residual | .834 | .517 | .400 4/4 |

Conclusions (B only, n = 2):
- **F1 confirmed on VAL oracle masks.** Adam's whole B gain lives in HF > 0.7 Nyquist: after LPF, B goes back to the V2 level (.472/.399 vs .469/.409). On A, LPF almost doesn't hurt (.929).
- **HF structure matters, not its power.** The same-power random HF (M3) costs −.12/−.16 B.
- **The HF can't be transplanted onto V2's LF.** M2 gives +.037/−.012 (mean +.012). That is below Gemini's own +.02 threshold and far from the oracle's +.23. Adam's HF is coherent only with its own LF: the solution is joint, not an additive "HF patch". By Gemini's rule this formally means "the deficit is not only in HF". More precisely: a usable HF component must be generated together with the LF, which matches the weak R18 effect (+.004).
- The one-step residual (M4) gives +.048/−.009. That is the same order as the bpmix per-quartet gain, and it is per-quartet physics, not a network.
- Gemini's G_xtalk metric grows for the oracle (.234 vs .185) as corr increases on all planes, so it is not a clean ghost measure; we don't use it.

Status: no new experiments are started; this is for #47 (defense) and the R18 conclusion.
