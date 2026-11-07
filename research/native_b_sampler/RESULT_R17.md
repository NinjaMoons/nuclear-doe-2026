# R17-BRECOMB result (IDEA_CARD f9239c5 gate) — **INCONCLUSIVE** (null effect)
init V2 19b05dad, 6000 updates, seeds 0/1, same idx sequence; recomb replaced 6144/12142 and 5928/11953 drawn B
quartets (p=.5) from the 118-quartet independent TRAIN pool (48 B64 + 70 B208-independent; VAL/TEST sha guard OK).
Seed mean, VAL16 official path, recomb - control:
| step | dP [CI] | dP_B [CI] | dSSIM | d worst | d orgsets10 | d WC64 |
|---|---|---|---|---|---|---|
| 2000 | +.0005 [-.0004,+.0017] | +.0005 | -.0004 | +.0016 | +.0017 | +.0024 |
| 4000 | -.0034 [-.0058,-.0007] | -.0032 | -.0063 | -.0077 | -.0115 | -.0078 |
| 6000 | +.0031 [+.0023,+.0039] | +.0032 [+.0024,+.0040] | +.0066 | +.0041 | -.0038 | +.0098 |
Step 6000 absolute: control P .6126 / B .4516; recomb P .6157 / SSIM .5596 / B .4547 / B SSIM .3691.
B own-target 25/26 (control) vs 26/25 (recomb); B inter-z .689/.686 vs .677/.684 (targets .05).
Gate dP >= +.01 not met, orgsets10 worse, sign flips between 4000 and 6000 -> INCONCLUSIVE; no promotion.
Wall ~100-108 s per arm.
