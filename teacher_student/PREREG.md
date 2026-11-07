# R16 teacher -> student distillation — pre-registration (2026-11-07 ~20:20 UTC). NOT STARTED.

Owner @playerr456 (#40). Host: Windows laptop CPU i5-13420H, no CUDA. Review requested in #49 (@ggmaster913).
Rule constraints: per-input optimisation at INFERENCE is forbidden (organizer); classical methods for DATA are allowed.
The teacher is used only to create TRAIN labels; the contest output stays one forward of the trained student.

## Hypothesis
Student STEM v2 (`19b05dad…083f`, VAL 0.6124/0.5536) underfits. Per-quartet phase-only Adam warm-started from the
student's own phase produces strictly better, physically realisable reconstructions on TRAIN quartets (measured: P
+0.10…+0.15 on 4 TRAIN quartets, `warm_probe.json`). Distilling these realisable reconstructions back into the
student should raise VAL quality, especially sharpness and contrast. It is NOT expected to cure B ghosting: teacher
inter-z stays ~ the same.

## Teacher labels (TRAIN512 only; VAL/TEST never touched)
- Init = student v2 phase (`infer_phase`); Adam lr 0.05, **50 steps**, objective = exact R3; best state by objective, wrapped.
- Store the teacher reconstructions `display(|U_teacher|)` per TRAIN quartet (float32, 512×4×128×128 ≈ 134 MB).
- Cost measured: 2.6–3.5 s/quartet → p50 ≈ 27 min, p90 ≈ 35 min CPU for 512 quartets.

## Losses (`teacher_student/losses.py`; tests `test_ts.py` 6/6)
- `L_opt = mean_{b,k,pixels} (display(A_student) − display(A_teacher))²` on the contract display `|U|/max|U|`, native grid.
  - Invariant to global piston, 2π wraps and any other phase giving the same |U|. **Chosen.**
- `L_circ = mean(1 − cos(φ_s − φ_t + c*))` with optimal piston c* removed. Only wraps/piston-invariant → **not used**.
  - Twin test: conjugate+flip is not an amplitude-equivalent solution in our chirped Fresnel (|ΔU| = 2.7), but `L_opt` is
    safer by construction.
- Student objective: `R3 + μ·L_opt`, **μ = 25 fixed by rule** μ = 0.5·mean R3 / mean L_opt on the 4 probe TRAIN quartets
  (24.6 → 25), before any training. No μ sweep.

## Arms (matched; one process on this CPU)
- control: fine-tune from v2 with **R3 only**;
- treatment: fine-tune from v2 with **R3 + 25·L_opt**;
- same init weights, fresh Adam, seed 0 (same batches), lr 1e-3, batch 4, fp32, TRAIN512 manifest `76d2474d`;
- 100-step health gate (finite, VAL export works), then **1000 updates** each (≈12–15 min + ≈3 min eval per arm).
- Evaluation: official DOE.png uint8 → reload → 4 Fresnel → 4 PNG → reload, VAL16 only. Readouts: step 1000 + rule 6697684744.

## Decision (same rubric as R15 / #47)
- **GO** (research candidate v3, separate package only) if ΔP ≥ +0.01, ΔSSIM ≥ −0.005, B own-argmax not lower, and the
  predeclared visual samples (A64:010, B64:002, B64:012, B64:032) show sharper shapes without more ghosting or speckle.
- **NO_GO** if ΔP ≤ −0.005 or ΔSSIM ≤ −0.01 or visually worse; otherwise **INCONCLUSIVE**.
- A GO never replaces v1/v2 automatically; replication (seed 1) before any promotion.

## Budget (p50 / p90)
Teacher labels 27 / 35 min + 2 arms × (15 / 20 min) → **≈ 1.0 / 1.3 h CPU**. Must start before 02:00 MSK and end
before 04:00 MSK research cutoff; otherwise HOLD. Nothing runs before review OK in #49 (or coordinator GO).

## Known risks
- **Teacher overfit to the training target:** labels are per-quartet optima, and the student may memorise them. VAL is
  the only judge.
- **Ghosting:** teacher reconstructions keep similar inter-z; no cure for B ghosting is expected.
- **Speckle:** Adam teachers can add speckle. Watch the speckle and brightness metrics.

## Addendum before start (2026-11-07 ~21:03 UTC = 00:03 MSK), after R16_SCIENTIFIC_GO #49/6712551688
- Review fixes in code @ e718cfc7: TRAIN-only content-addressed label cache (labels = display of the uint8-codec teacher
  phase), alignment + permutation tests 4/4, speckle readout (`speckle.py`), student wrapper `train_r16.py`.
- Student init = v2 `19b05dad…083f`; teacher init = v2 phase; teacher 50 steps; labels for ALL 512 TRAIN quartets.
- Both arms train on exactly the labelled TRAIN set (here: all 512), same seed 0 batch generator, 1000 updates, eval every 100
  (step 100 = health gate: finite + VAL export works, else STOP).
- Added NO_GO (#49 item 3): treatment hot-px (PNG>=0.9) or peak-pixel fraction > 1.5x control, or visibly grainier with dP < +0.01.
- Readouts additionally: speckle.py metrics, paired bootstrap over 16 quartets, best_step==0 fraction of the teacher.
- Host: Matvey laptop CPU only (no GPU). EXPERIMENT_ID=R16. Cap: labels 45 min, each arm 30 min; whole chain must end before 03:30 MSK.
