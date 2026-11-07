# R18 BP-mix head — RESULT seed 0 (pre-registered PLAN_R18.md), 08.11.2026 ~01:35 MSK
Verdict: **INCONCLUSIVE(+)** — small, consistent gain, below the pre-registered GO threshold (dP >= +.01). Not promoted.
Host Mac M1 CPU, 4 threads; init v2 19b05dad; 6000 updates each arm; sampler randperm(512, gen(30000+seed)); R3; VAL16.
| arm | P | SSIM | A P/S | B P/S | B own | B margin | B inter-z full/low/band | worst P/S | mask HF | hot | top |
|---|---|---|---|---|---|---|---|---|---|---|---|
| control (R3 fine-tune) | .6112 | .5518 | .7712/.7416 | .4513/.3619 | 26/32 | .098 | .687/.939/.605 | .593/.504 | .137 | .0004 | .0011 |
| bpmix | .6154 | .5560 | .7761/.7434 | .4546/.3685 | 26/32 | .100 | .677/.936/.591 | .593/.503 | .154 | .0004 | .0012 |
Paired bootstrap (2000, over quartets) bpmix - control: dP +.0041 [+.0028, +.0055] (16/16 quartets better),
dSSIM +.0042 [+.0013, +.0068] (14/16), dB_margin +.0018 [-.0016, +.0052], dB_own 0. Reference v2: .6124/.5536.
Visual (#47 fixed A64:060, B64:012, B64:016): indistinguishable from control; B ghosting unchanged.
Mechanism (VAL16, bpmix ckpt): head strongly used — mean |c0| .71, mean |a_k| .20, |sum a_k B_k| .90, output phase
differs from the STEM branch by mean 1.83 rad — yet the OUTPUT mask stays smooth (HF r>.5: .073 vs control .137 is
the evaluator's per-quartet mask HF; teacher masks .6-.7). Training reorganises the smooth solution; it does not learn
the teacher-like target-specific HF masks.
Comparison: R8-RES (global 1x1 mixing, #43): dP +.0002 [-.0023,+.0020]; R18 (spatially adaptive smooth maps): +.0041
[+.0028,+.0055] — direction as predicted (adaptive > global), magnitude ~10x smaller than the per-quartet ceiling
(+.06..+.13 over v2 on two TRAIN-B quartets).
Scientific conclusion (seed 0; seed 1 running): physically structured HF (B_k) makes per-quartet optimisation reach
~60% of the free-phase gain, but a single shared network trained with R3 converts only a small part of that into VAL
quality; the gap is amortisation of target-specific high-frequency phase, not the optics and not a missing input.
Checkpoints: results/CHECKPOINT_SHA256.txt (local, not committed). Seed 1 replication result will be appended.

---
# Seed 1 replication and seed-mean verdict, 08.11.2026 ~02:25 MSK
Same host, script and init; sampler gen(30001); 6000 updates per arm.

| arm | P | SSIM | A P/S | B P/S | B own | B margin | B inter-z full/low/band | worst P/S | mask HF |
|---|---|---|---|---|---|---|---|---|---|
| control s1 | .6130 | .5579 | .7735/.7498 | .4526/.3659 | 26/32 | .100 | .677/.938/.593 | .595/.513 | .145 |
| bpmix s1 | .6135 | .5544 | .7741/.7448 | .4528/.3639 | **24/32** | .103 | .678/.938/.598 | .599/.509 | .139 |

Seed 1 paired, bpmix − control: dP +.0004 [−.0004, +.0014] (10/16), dSSIM **−.0035** [−.0079, −.0004] (4/16), dB own −2/32.

**Seed mean** (s0, s1; per-quartet means, paired bootstrap 2000), `results/r18_seedmean.json`:
- dP +.0023 [+.0015, +.0032] (16/16); dSSIM +.0004 [−.0030, +.0028];
- dB_P +.0018 [+.0011, +.0025]; dA_P +.0028; dB margin +.0025 [+.0006, +.0047];
- dB own −.125 per quartet (s0 26=26, s1 24<26); worst plane P .5960 vs .5941.
- References: v2 .6124/.5536; R8-RES (#43) dP +.0002.

Visual (s1 sheets `results/r18_s1_sheet_*.png`): indistinguishable from control. B64:012 ghosting is the same (cross-plane Pearson .18–.37 in both).

## Final verdict R18: **NO_GO (negligible).**
- The seed-0 gain of +.004 did not replicate (s1 +.0004).
- The seed mean of +.002 P is ~5× below the pre-registered gate.
- SSIM and B own do not improve.

## Scientific conclusion (for #47)
- **Per-quartet optimisation:** a physically structured HF path (complex mixing of exact back-projections B_k) reaches ~60% of the free-phase Adam gain on TRAIN-B quartets.
- **Trained network:** the same path inside a network trained with R3 gives +.002 P on VAL and no change in ghosting. The network uses the head heavily (|Σa_k B_k| .90, Δφ 1.83 rad), yet the output masks stay smooth (HF .07–.15 vs teacher .6–.7).
- **G4 M0–M4 test:** coherent HF only works together with its own LF (a V2 LF + oracle HF transplant gives +.012 vs +.23 for the oracle).

Physically structured HF therefore helps **per-quartet optimisation, not the trained one-forward network**. The B deficit is the amortisation of target-specific coherent HF phase, not optics, not a missing input and not head capacity. Per-quartet ceilings must not be reported as contest-model quality.
