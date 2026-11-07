# R8-RES result (PLAN_R8RES.md) — **INCONCLUSIVE (null)**
Seed mean vs matched R17 control, VAL16 official path. Step 6000: P .6128 (d +.0002 [CI -.0023,+.0020]), SSIM .5561
(+.0031), B .4529 (+.0013), B SSIM .3656 (+.0019), worst +.0005, orgsets10 +.0017, WC64 -.0002; step 4000 d -.0032.
B own-target 26/24 vs 25/26; B inter-z .680/.689 vs .689/.686. Gate not met.
Mechanism check (VAL16): the phasor head IS used (|w| .84/.88, bias (.88,-.11)/(.89,-.10), mean |atan2 term| .16-.17
rad, p99 .43-.44) but (re, im) stays far from the origin (min re .53-.59) -> no phase singularities; added-term
roughness .08 rad/px. Gradient descent on R3 from the smooth V2 basin does not use the extra expressivity: the
limit is the optimisation landscape of one-forward training, not head capacity. Wall ~100 s / arm.
