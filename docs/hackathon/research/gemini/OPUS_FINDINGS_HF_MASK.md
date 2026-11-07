# Opus findings (read-only, CPU), 08.11 ~00:30 MSK — where the missing B quality lives

All on TRAIN B64:010 / B208:006 (TRAIN only), official uint8 DOE export path, R3 objective; scripts in tests/.

## F1 (CONFIRMED) Teacher quality is carried by mask high frequencies
`lpf_mask_test.py`: low-pass exp(i*phi) of masks (radial cutoff rc in Nyquist units), arg, official export.
| mask | rc | power kept | P (mean 4 planes) | inter-z (phys grid) |
|---|---|---|---|---|
| Adam1000 (#14) B208:006 | none / 0.7 / 0.5 / 0.3 | 1 / .40 / .21 / .07 | .661 / .399 / .314 / .216 | .40 / .49 / .63 / .79 |
| Adam1000 B64:010 | none / 0.7 / 0.5 / 0.3 | 1 / .42 / .23 / .09 | .562 / .334 / .286 / .232 | .53 / .54 / .66 / .81 |
| v2 B208:006 | none / 0.7 / 0.5 / 0.3 | 1 / .87 / .79 / .64 | .461 / .444 / .423 / .362 | .62 / .66 / .72 / .84 |
| v2 B64:010 | none / 0.7 / 0.5 / 0.3 | 1 / .97 / .96 / .92 | .427 / .419 / .413 / .396 | .81 / .82 / .85 / .89 |

## F2 (CONFIRMED) A warm teacher from v2 gains P only by adding HF, and the gain vanishes under low-pass
`warm_teacher_spectrum.py` (Adam lr .05 from the v2 phase; best state; official export):
| quartet | step | P | own | HF(r>.5) | R90 | P after LPF rc .7 |
|---|---|---|---|---|---|---|
| B208:006 | 0 / 50 / 300 / 1000 / 2000 | .461 / .606 / .655 / .665 / .667 | 4/4 | .21 / .46 / .64 / .68 / .69 | .78 / 1.00 / 1.07 / 1.08 / 1.08 | .444 / .502 / .464 / .453 / .449 |
| B64:010 | 0 / 50 / 300 / 1000 / 2000 | .427 / .524 / .561 / .570 / .572 | 3/4 | .04 / .29 / .51 / .57 / .59 | .27 / .93 / 1.02 / 1.05 / 1.05 | .419 / .452 / .422 / .411 / .404 |
Reading: the network's masks are smooth (most power below 0.7 Nyquist); the teacher's extra fidelity lives at
1-3 px mask scales. Consistent with neutral width/distillation and R12 (a random HF basis is not the needed HF).
Inter-z (index grid) does not drop for the teacher — the gain is own-target fidelity, not plane separation.

## F3 (CONFIRMED per-quartet ceiling, HYPOTHESIS for a trained net) Physically structured HF path
`bpmix_ceiling.py`: phi = arg(c0*exp(i*phi_v2) + sum_k c_k*B_k), B_k = STEM back-projection features, c complex on a
smooth 32x32 grid (bilinear), per-quartet Adam on R3, 1000 steps:
| quartet | v2 | BP-mix (v2+bp) | BP-mix (bp only) | control: same 5 smooth complex maps, no B_k | control: c0 only | free warm Adam 2000 |
|---|---|---|---|---|---|---|
| B208:006 P | .461 | **.593** | .584 | .533 | .498 | .667 |
| B64:010 P | .427 | **.512** | .504 | .483 | .460 | .572 |
(first-run JSON of the BP-mix rows was overwritten by the control run; values from the run log above.)
=> B_k-structured HF adds +.03..+.06 over equal-capacity smooth correction and reaches ~60% of the free-phase gain.
-> IDEA_CARD R18 (BP-mix head), pilot running on Mac CPU (branch research/r18-bpmix-head).
