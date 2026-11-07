# R8-SCRATCH result — **INCONCLUSIVE (positive trend)** · and pre-registered STACK (2026-11-08 ~01:00 MSK)
R8-SCRATCH (phasor head from scratch, W16, 12k, const lr) vs W16 12k control L_native, seed mean, VAL16 official:
step 12000 dP +.0083 [CI +.0065,+.0103], dSSIM +.0159, dB +.0075, dB_SSIM +.0094, d worst +.0064, d orgsets10
+.0154, d WC64 +.0025; step 8000 dP +.0108 [+.0090,+.0128]. Absolute 12k: P .6168 / SSIM .5655 / B .4566
(s1 .6189 / .5710). B own 25/25 vs 25/25; B inter-z .694/.686 vs .687/.669 (ghosting unchanged).
Gate dP >= +.01 missed by .0017 at the primary step -> INCONCLUSIVE; first factor with a consistent plus on all sets.
The basin explanation is partly supported: the same head from scratch gives +.008, from V2 +.0002.

## STACK (pre-registered before running) — the 3 positive-trend factors together
PhasorStemUNet from scratch (phasor N(0,.5), bias 0) + base_channels 32 + cosine lr 1e-3->1e-5, 24000 updates,
sampler randperm(512, gen(seed))[:4], R3, batch 4, seeds 0/1, ckpts every 4000. Compared with V2 19b05dad
(.61243/.55356/B .4516) and with W32 cosine 24k (RESULT_RECIPE.md, same sampler/steps, only the head differs).
Promotion gate vs V2 (primary step 24000 seed mean): dP >= +.01, dB >= +.005, dSSIM >= 0, B own not lower,
orgsets10 and WC64 not worse. A pass makes it a RESEARCH CANDIDATE only: it needs the PhasorStemUNet class in the
package (loader change) -> CPU CLI + package review by #8/@ggmaster913; V2 stays protected. Wall p50/p90 25/35 min.

## STACK result — **NOT PROMOTED** (gate failed: dP +.0071 < +.01, WC64 worse, B own s0 lower)
Seed mean vs V2 at step 24000: P .6196 (+.0071; s0 +.0064 [+.0042,+.0087], s1 +.0079 [+.0051,+.0107]), SSIM .5694
(+.0158; s1 16/16 quartets), A .7784 (+.0052), B .4607 (+.0091), B SSIM .3743 (+.0093), worst .6030 (+.0114),
orgsets10 .5271/.3420 (+.0171/+.0203), WC64 .3709 (-.0149). vs W32 cosine 24k (same everything but the head):
dP +.0021, dSSIM +.0070, dB +.0038. B own 24/26 (V2 25), A own 31/31 (V2 32), B inter-z .691/.687 (ghosting
unchanged). Peak 16000: P .6213 / SSIM .5709 (reported only; primary = 24000). Best single: s1 24k sha fb9ba7bc…
P .6203 / SSIM .5718 / B .4617 — research checkpoint only (needs PhasorStemUNet + infer_phasor.py; CPU one-forward
CLI verified on R8-SCRATCH). #43 research closed at 01:10 MSK per plan; V2 stays the protected model.
