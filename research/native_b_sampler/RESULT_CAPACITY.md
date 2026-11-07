# Capacity result (pre-registered in RESULT_B75_ORACLE.md) — **CAPACITY_NEUTRAL**
Seed mean (0,1), step 12000, VAL16 official path; control W16d3 = L_native_s0/s1 (P .6085 / SSIM .5496 / B .4491).
| arm | P | SSIM | B P | B SSIM | dP [CI] | dB [CI] | d orgsets10 P | d WC64 P |
|---|---|---|---|---|---|---|---|---|
| W32d3 | .6156 | .5590 | .4547 | .3668 | +.0071 [+.0056,+.0089] | +.0057 [+.0044,+.0070] | +.0124 | +.0075 |
| W64d3 | .6150 | .5631 | .4558 | .3721 | +.0065 [+.0039,+.0091] | +.0068 [+.0048,+.0087] | +.0063 | -.0002 |
| W32d4 | .6131 | .5579 | .4537 | .3652 | +.0047 [+.0016,+.0074] | +.0047 [+.0035,+.0058] | +.0059 | +.0034 |
No arm reaches the fixed dP >= +.01 gate. Versus the V2 research candidate 19b05dad (single checkpoint, P .6124,
B .4516) the best arm W32d3 is +.003 P / +.003 B: not a promotion. Width gives a small, consistent gain; the B
gap to the oracle (B .6526) stays. Train time 4.7-5.9 min per 12000-update run on the 4070 Ti.
