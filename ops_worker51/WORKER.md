# #51 GPU worker procedure (DESKTOP-T1K698V, RTX 4070 Ti) — read on every non-empty tick

1. `C:\nuclear-doe-operator\desktop-41\.venv-41\Scripts\python.exe -I C:\nuclear-doe-operator\ops\worker51\precheck.py`
   - NO_TASK / LOCKED -> answer one line `worker51 tick <now_msk>: <action>` and STOP (no other tool calls).
   - BUSY -> a GPU python job is running: post nothing, STOP (one GPU job per host).
   - GH_ERROR -> one line, STOP.
2. TASK: open the card (url). Execute ONLY if ALL hold, else comment `TASK_ID=<id> STATUS=BLOCKED | reason` in #51
   and record it as processed:
   - fields present: TASK_ID, TO=@MVRto01/DESKTOP-T1K698V, OWNER/APPROVAL, GOAL, INPUT_REFS/SHA, SOURCE_SPLIT+RIGHTS,
     PRE-REG SCIENCE_GATE, ETA_P90, OUTPUT_DIR/BRANCH, ACCEPTANCE, STOP/FAIL RULES, STATUS=READY;
   - author NinjaMoons (coordinator) or ggmaster913 with an explicit science GO;
   - data = native TRAIN512 (manifest 76d2474d) or other data with written rights; no VAL tuning, TEST never decoded;
   - team gates MSK (coordinator #51/6714140160, supersedes 02:00/04:00): `phase` from precheck:
     TRAIN_ALLOWED_WITH_GO 02:00-04:30 = bounded, preregistered, scientifically distinct TRAIN-only training with a
     matched control, only with @ggmaster913 `SCIENCE_GATE=GO` or explicit coordinator GO, ETA_P90 must end by 05:15;
     FINISH_ONLY 04:30-05:15 = no new run, finish/export/compare; ACCEPTANCE_ONLY 05:15-06:00 = acceptance and
     packaging checks only; FROZEN after 06:00 = no model changes;
   - not a repeat of R16/R17/R18/R8/B75/capacity/distill/recipe/STACK; no endless VAL16 tuning;
   - not a repeat of a finished experiment (see b-sampler/research/native_b_sampler/NIGHT_SUMMARY.md, RESULT_*.md);
   - never modify V1/V2/4ch packages or checkpoints; never official submission; no paid actions; no public ports.
3. Claim: comment `TASK_ID=<id> STATUS=CLAIMED | HOST=DESKTOP-T1K698V GPU-282fb2b9 | branch | out dir | ETA`.
   Add to `C:\nuclear-doe-operator\ops\worker51\state.json` processed[<id>] = "CLAIMED".
4. Work in a NEW git worktree from origin (branch named in the card, else `research/sanya-w51-<id>`), outputs under
   `C:\nuclear-doe-operator\desktop-41\w51\<id>\`. Start GPU jobs with Bash run_in_background; one at a time.
   Delete `C:
uclear-doe-operator\ops\worker51	ick.lock` when the task is finished, blocked or failed.
5. Publish: comment `TASK_ID=<id> STATUS=DONE|NO_GO|FAILED` with metrics (official DOE u8 path, A/B/worst, B own),
   checkpoint/source SHA, private asset id if any, wall time, NEXT. Set processed[<id>] accordingly. Update
   `C:\nuclear-doe-operator\b-sampler\NEXT_STEP.md` and push.
