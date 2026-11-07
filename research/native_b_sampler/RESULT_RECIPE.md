# RECIPE result (pre-registered in RESULT_DISTILL.md) — **RECIPE_NEUTRAL** (no promotion over V2)
W32d3, native TRAIN512, R3, batch 4, seeds 0/1, 24000 updates; seed mean, VAL16 official path.
| | P | SSIM | A P | B P | B SSIM | worst | orgsets10 P/S | WC64 P |
|---|---|---|---|---|---|---|---|---|
| V2 19b05dad (W16, 12k const) | .6124 | .5536 | .7733 | .4516 | .3650 | .5917 | .5100/.3217 | .3858 |
| W32 const 24k | .6156 | .5576 | .7772 | .4540 | .3642 | .5969 | .5236/.3293 | .3811 |
| W32 cosine 24k | .6175 | .5624 | .7780 | .4570 | .3706 | .6004 | .5222/.3318 | .3813 |
Cosine vs V2 per quartet: s0 dP +.0056 [+.0040,+.0077] 16/16, dSSIM +.0111; s1 dP +.0045 [+.0027,+.0062] 15/16.
Cosine vs const: dP +.0019 [-.0012,+.0041]. Curves plateau ~.618 from 16k on (const 20k .6182 is the max).
Gate dP >= +.01 vs V2 NOT met (+.005) and WC64 slightly worse -> V2 stays. The W16/12k V2 is within ~.005 of what
this architecture + R3 reaches with 2x width and 2x updates. Wall: 8.2-8.8 min per 24k run on the 4070 Ti.
