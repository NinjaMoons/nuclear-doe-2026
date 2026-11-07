# Back-Projection Stem — PREP report (research only)

Status: PREP_DONE (stem technically ready for review). GPU PILOT AUTHORIZED? NO.

- Base: 49b96041b3b4bf1b2eaa57567939bcd929defd2e; only `back_projection_stem/` added on a local branch.
- Device: torch 2.9.0+cpu, `torch.cuda.is_available() = False`. Only an Intel UHD Graphics adapter, no NVIDIA. Therefore CPU tests; VRAM = PENDING.
- Tests: `python -m unittest back_projection_stem.test_stem -v` -> 11/11 OK (`unittest_log.txt`). The production best.pt strict-load check ran with `BPS_BEST_PT`.
- Round trip, rel L2 max over 4x6, 5x7 (dy 10 um != dx 13 um), 128x128 x 4 official z, B=2..3, both directions:
  - complex64: 2.1e-07;
  - complex128: 4.3e-16.
- Independent reference: direct-sum DFT of the continuous formula equals the official forward (< 1e-12). The analytic inverse equals `numpy.linalg.solve` (< 1e-10), and `Minv @ M ~ I` (< 1e-10) on 4x6 / 5x7.
- Step cost (CPU fp32, 8 threads, batch 4, 128x128, R3 loss, forward+Fresnel+loss+backward+Adam):
  - 4-ch: 0.235 s;
  - 12-ch: 0.249 s, ratio 1.06 in interleaved A/B rounds;
  - round-to-round spread +-0.04 s is larger than the difference;
  - an earlier non-interleaved run gave 1.71, an order/thermal artifact, superseded.
- Features: 0.027-0.035 s per batch. They depend only on the targets and could be precomputed once per dataset.
- Params: 451 329 -> 452 481 (+1 152, first conv only).
- Not done: no training of any length, no TRAIN/VAL/TEST change, no production file touched.
