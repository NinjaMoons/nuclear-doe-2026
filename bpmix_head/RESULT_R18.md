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
