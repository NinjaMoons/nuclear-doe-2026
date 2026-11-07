# #43 night summary — @MVRto01 desktop RTX 4070 Ti (07→08 Nov, research stopped 01:10 MSK)
All numbers: VAL16 (8 A + 8 B), official path DOE.png u8 -> reload -> 4x Fresnel -> 4 PNG -> reload; seed mean (0,1);
TEST never decoded; native TRAIN512 only (WhiteCollars only in the #41 pilot, weights kept local).

| experiment | factor | P | SSIM | B P | verdict |
|---|---|---|---|---|---|
| #40 V1 569e1700 | STEM 12ch R3, 900 upd CPU | .5778 | .5067 | .4257 | reference |
| **V2 19b05dad** | same, 12000 upd GPU | **.6124** | **.5536** | **.4516** | LONGER_HELPS -> protected V2 |
| WC pilot (4000 upd) | + WhiteCollars 50/50 | -.0065 vs native | -.0081 | -.0062 | WC_NEUTRAL (dropped) |
| B75 | A25/B75 sampler, from V2 | -.0034 | -.0059 | -.0020 | NEUTRAL |
| capacity | W32d3 / W64d3 / W32d4, 12k | +.007 / +.007 / +.005 vs W16 | +.009..+.014 | +.005..+.007 | NEUTRAL |
| distill | phase teacher (Adam2000, TRAIN) | -.0059 | -.0013 | .0000 | NEUTRAL |
| recipe | W32 24k cosine vs V2 | +.0051 | +.0088 | +.0054 | NEUTRAL (gate +.01) |
| R17 | independent B recombination | +.0031 | +.0066 | +.0032 | INCONCLUSIVE |
| R8-RES | phasor-residual head | +.0002 | +.0031 | +.0013 | INCONCLUSIVE |
| Adam oracle (diagnostic) | 2000 Adam steps from V2 phase | .7952 | .7415 | .6526 (own 31/32) | NOT a contest method |

Why B ghosting persists (for #47):
- Physics is NOT the limit: the same single phase-only 8-bit DOE, optimised per input, separates B planes (B own
  31/32, B inter-z .50 vs V2 .69; raw sheet `sheet_TARGET_V2_ORACLE_A060_B012_B016.png`).
- The network underfits (TRAIN64 P .620 ~ VAL .612): more data (WC), more width, longer/cosine schedules, B-focused
  sampling, B decorrelation and a phasor head all move P by <= +.007; the one-forward U-Net trained by R3 converges to
  a smooth-phase basin (G1: roughness .07-.42 vs oracle 3.0); a phase-space teacher is not learnable (distance .58
  -> .48), and the extra phasor head stays far from phase singularities.
- Honest claim: "one forward pass, no optimiser at inference; ~.61 Pearson; B shows cross-plane ghosting that is a
  limitation of the learned inverse, not of the optics" — supported by the oracle sheet above.
Best numeric research checkpoint (not promoted, gate not met): W32 cosine 24k s0 P .6180 / SSIM .5649.
