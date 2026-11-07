# R16 NEXT_STEP (written 2026-11-07T21:36Z, host = Matvey laptop, CPU only)

**Running.** Chain started 00:17 MSK as a Claude Code background task from this session (bash, run_in_background). It is
**not** an independent OS scheduler job, so keep this Desktop session and the laptop awake until done.
- python PIDs at writing: 19932  12636  
- Stages: (1) teacher labels → C:/Users/user/AppData/Local/Temp/r16runs/labels512.pt (log labels.log);
  (2) control arm → r16runs/control (control.log); (3) treatment arm → r16runs/treatment (treatment.log).
- Caps: labels 45 min, each arm 30 min (timeout). Hard research stop: 04:00 MSK.

**If the session died before completion:** look at r16runs/*.log. Do NOT restart blindly.
- If labels512.pt exists but an arm is missing, rerun only that arm, from the r16 worktree, same command, NEW out-dir:
  `python -m teacher_student.train_r16 --arm <control|treatment> --init <v2 ckpt 19b05dad> --labels r16runs/labels512.pt --manifest c4b/manifest_expanded.json --out-dir r16runs/<arm>_rerun --device cpu --updates 1000 --eval-every 100 --seed 0 --threads 8`
- If it is past ~02:30 MSK and both arms do not fit before 04:00, call **R16 NO_GO (time)** and post evidence in #49.

**Readout when both arms are done** (scratchpad tools):
1. `python r3/r3_compare.py r16runs/compare.json ctl=r16runs/control treat=r16runs/treatment` → VAL16 P/SSIM, A/B, per-z, worst, deltas per step; plus rule 6697684744.
2. `python -m teacher_student.speckle r16runs/speckle.json ctl=<control step1000 val export> treat=<treatment step1000 val export>`.
3. Separation (own-argmax, inter-z physical) + paired bootstrap; visual sheet on A64:010, B64:002, B64:012, B64:032.
4. Decision per PREREG (+ addendum NO_GO speckle rule). Post R16 result in #49, request @ggmaster913 audit; upload checkpoint + results to the private release if GO.
