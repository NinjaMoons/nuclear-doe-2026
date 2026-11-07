# MVR research journal (#6, autonomous R&D per #7/6705970325)

Base: `49b96041b3b4bf1b2eaa57567939bcd929defd2e` (frozen training tree), branch `codex/6-mvr-research` (never merged).
Data: release `data-4b-v1` / `corpus-4b-win-49b9604.tar.gz` (sha256 6c6f136b…5701), manifest_expanded 76d2474d…6f1c78.
Splits: TRAIN 512 / VAL 16. TEST quartets are read and hashed by `load_manifest` (as in run1000) but never
reach model, loss, selection or evaluation. Production checkpoint, source, TEST untouched.

## E1 — λ control vs treatment (pre-registered before any VAL number was seen)
- Hypothesis: raising the contract-MSE weight λ in `lsq_nmse + λ·contract_mse` from 0.1 to 0.3 changes VAL
  global Pearson/SSIM of a short run. Research hypothesis only, no prediction of improvement.
- Arms: λ=0.1 (control) and λ=0.3 (treatment), both through `research/mvr/research_lambda_wrapper.py`.
- Fixed: `--updates 200 --eval-every 100 --batch-size 4 --lr 0.001 --seed 0 --base-channels 16 --depth 3`,
  default objective, fp32, same seed ⇒ identical init and batch order. Fresh init (no run1000 checkpoint).
- Device: **CPU** (RTX 4050 not visible to Windows: dGPU off / no NVIDIA display driver — GPU_BLOCKED).
  Same device and threads for both arms; timings are CPU and not comparable with run1000 GPU.
- Primary readout: VAL export metrics of `last.pt` (step 200) — global Pearson, global SSIM, A/B, per-plane,
  worst-plane — from the existing new-process VAL evaluator. Secondary: best.pt by own objective;
  lsq_nmse and contract_mse from train/val logs (raw objective is not comparable across λ).
- Compute cap: decided after a 5-step CPU smoke; abort if one arm > 30 min.
- Decision rule: n_val=16 → descriptive only. Call a direction only if both Pearson and SSIM move the same
  way on ALL and on at least one of A/B by > 0.01; otherwise "no detectable effect at 200 updates".

### E1 result (2026-11-07 11:15 UTC, CPU, 6 threads, ~54 s train + VAL eval per arm)
best.pt = last.pt = step 200 in both arms (selection by own objective did not differ from last).

| VAL (16: 8A+8B), step 200 | λ=0.1 | λ=0.3 | Δ(0.3−0.1) |
|---|---|---|---|
| Pearson ALL / A / B | 0.3678 / 0.4951 / 0.2404 | 0.3641 / 0.4936 / 0.2345 | −0.004 / −0.002 / −0.006 |
| SSIM ALL / A / B | 0.2663 / 0.4127 / 0.1199 | 0.2820 / 0.4258 / 0.1381 | +0.016 / +0.013 / +0.018 |
| worst-plane mean P / S | 0.3356 / 0.2583 | 0.3422 / 0.2733 | +0.007 / +0.015 |
| worst sample-plane P | B64:057 z1 −0.117 | B64:057 z1 −0.011 | |
| VAL contract_mse | 0.0871 | 0.0377 | −57 % |
| VAL lsq_nmse = obj − λ·contract | 0.3938 | 0.3929 | ≈0 |
| lit_frac (target 0.108) | 0.242 | 0.047 | over- vs under-exposed |

Sanity: CPU control at step 200 (P 0.3678, S 0.2663) ≈ run1000 GPU step_000200 (P 0.3725, S 0.2628).
Verdict by the pre-registered rule: **no detectable effect** (Pearson and SSIM do not move together by >0.01).
Observation: λ mostly moves the output brightness distribution (lit_frac 0.24 → 0.05 vs target 0.11) and
contract MSE, not LSQ-NMSE or Pearson; SSIM +0.016 is within what a single seed can show.

## E2 — does the λ effect survive the full 1000-update budget and seed noise? (pre-registered)
- Arms λ=0.1 and λ=0.3, seeds 0 and 1 (4 runs), `--updates 1000 --eval-every 100`, everything else as E1, CPU.
- Primary: VAL last.pt (step 1000); secondary: best.pt by own objective; lit_frac; contract_mse/lsq_nmse.
- Noise floor: |Δ seed0 − Δ seed1| for each metric. Call an effect only if Δ has the same sign in both seeds
  and |mean Δ| > 0.01 on ALL for SSIM or Pearson, without the other metric dropping by > 0.01.
- Compute cap: ~6 min per run, 40 min total; abort otherwise. Stop all research by 19:00 MSK.

### E2 result (2026-11-07 11:38 UTC, CPU, ~225 s train + VAL eval per run; best.pt = last.pt = step 1000 in all 4)

| VAL step 1000 | λ=0.1 s0 | λ=0.3 s0 | λ=0.1 s1 | λ=0.3 s1 | mean Δ(0.3−0.1) |
|---|---|---|---|---|---|
| Pearson ALL | 0.4174 | 0.4019 | 0.4283 | 0.4036 | **−0.020** (−0.016, −0.025) |
| Pearson A / B | 0.5434 / 0.2913 | 0.5315 / 0.2724 | 0.5557 / 0.3009 | 0.5404 / 0.2668 | A −0.014, B −0.027 |
| SSIM ALL | 0.3312 | 0.3141 | 0.3329 | 0.3211 | **−0.014** (−0.017, −0.012) |
| SSIM A / B | 0.4811 / 0.1813 | 0.4633 / 0.1650 | 0.4866 / 0.1792 | 0.4846 / 0.1576 | A −0.010, B −0.019 |
| worst-plane mean P / S | 0.3733 / 0.3137 | 0.3814 / 0.2865 | 0.3979 / 0.3071 | 0.3734 / 0.3101 | P −0.008, S −0.012 |
| VAL contract_mse | 0.0401 | 0.0380 | 0.0487 | 0.0327 | |
| VAL lsq_nmse | 0.3700 | 0.3755 | 0.3693 | 0.3773 | +0.007 (worse) |
| lit_frac (target 0.108) | 0.087 | 0.059 | 0.112 | 0.051 | under-exposed at λ=0.3 |

Seed noise of the control: |P s0−s1| = 0.011, |S s0−s1| = 0.002.
Sanity: CPU control s0 step 1000 (P 0.4174, S 0.3312) vs run1000 GPU best.pt (P 0.4180, S 0.3215).
Verdict by the pre-registered rule: **effect, λ=0.3 is worse** than λ=0.1 at 1000 updates (same sign in both
seeds, |mean ΔP| = 0.020 > 0.01, SSIM also down). E1's SSIM +0.016 at 200 updates did not persist.
NOGO for λ=0.3. Mechanism (descriptive): larger λ pushes outputs dimmer (lit_frac 0.05–0.06 vs target 0.11)
and worsens LSQ-NMSE, the part that tracks Pearson.

## E3 — is λ below 0.1 better? (pre-registered before running)
- Direction from E2: smaller contract weight was better. Test λ=0.03 and λ=0.0 (pure LSQ-gain NMSE),
  seeds 0 and 1, 1000 updates, everything else as E2. Controls = E2 λ=0.1 runs (same code/device/seeds).
- Same rule as E2 per arm vs control. Watch lit_frac (λ=0 removes the only brightness anchor).
- Cap: 4 runs × ~6 min; stop research by 19:00 MSK.

### E3 result (2026-11-07 11:59 UTC, CPU; primary = last.pt step 1000; controls = E2 λ=0.1)

| VAL step 1000 | λ=0.1 s0 / s1 | λ=0.03 s0 / s1 | λ=0.0 s0 / s1 |
|---|---|---|---|
| Pearson ALL | 0.4174 / 0.4283 | 0.4357 / 0.4336 | 0.4225 / 0.4332 |
| Pearson A | 0.5434 / 0.5557 | 0.5691 / 0.5610 | 0.5475 / 0.5697 |
| Pearson B | 0.2913 / 0.3009 | 0.3022 / 0.3062 | 0.2976 / 0.2967 |
| SSIM ALL | 0.3312 / 0.3329 | 0.3341 / 0.3270 | 0.3011 / 0.3248 |
| SSIM A / B (s0) | 0.4811 / 0.1813 | 0.4815 / 0.1867 | 0.4415 / 0.1606 |
| worst-plane mean P | 0.3733 / 0.3979 | 0.4029 / 0.4093 | 0.3789 / 0.4097 |
| VAL lsq_nmse | 0.3700 / 0.3693 | 0.3637 / 0.3669 | 0.3694 / 0.3650 |
| lit_frac (target 0.108) | 0.087 / 0.112 | 0.212 / 0.210 | 0.340 / 0.240 |

Δ vs control (mean of seeds): λ=0.03 ΔP **+0.012** (+0.018, +0.005; A +0.016, B +0.008), ΔS −0.002
(+0.003, −0.006), worst-plane P +0.021. λ=0.0 ΔP +0.005, ΔS **−0.019**.
Verdict by the rule: λ=0.03 passes narrowly (same sign both seeds, |ΔP| 0.012 > 0.01, SSIM not down > 0.01)
— but the margin is about the control seed spread (0.011), so it is a **candidate, not a confirmed gain**.
λ=0.0: no (SSIM drops, outputs over-exposed). λ=0.03 run s1: best.pt was step 900 (by own objective);
primary readout stays last.pt as pre-registered.
Overall picture: VAL Pearson is maximised around λ≈0.03–0.1 and falls at λ=0.3; λ trades brightness
(lit_frac) against contract MSE; SSIM peaks near λ=0.03–0.1.

## Decision
Autonomous budget (1 experiment + 2 follow-ups, #7/6705970325) is used: E1, E2, E3. NEXT (needs coordinator
GO, not started): confirm λ=0.03 vs 0.1 with ≥3 more seeds on the same device (GPU, once the driver is installed),
VAL only. Production best.pt stays λ=0.1; nothing here changes the contest model.
