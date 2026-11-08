# NEXT_STEP — @MVRto01 desktop Claude (#43 / #51) — updated 2026-11-08 03:05 MSK

Read this first after any quota reset / restart. Then `git fetch origin` and re-read #43, #49, #51, #8 and
`git show origin/main:docs/hackathon/NIGHT_AGENT_PROTOCOL.md`. Never restart a job whose outputs already exist.

## Hosts
- HOST1 = DESKTOP-T1K698V, RTX 4070 Ti 12 GB, GPU-282fb2b9-94df-8922-4941-d3b30604ee5f, driver 616.56,
  venv `C:\nuclear-doe-operator\desktop-41\.venv-41` (py3.11.9, torch 2.9.0+cu128). ONE GPU job at a time.
- HOST2 = Sanya laptop: **UNVERIFIED** (no hostname/GPU known; no SSH server reachable on LAN 192.168.1.0/24, desktop
  has no SSH keys). Mode B (separate Claude Code on the laptop + GitHub locks) unless the owner enables OpenSSH there.
  Bootstrap for it: `research/native_b_sampler/LAPTOP_BOOTSTRAP.md`.

## State on HOST1
- GPU IDLE. #43 research CLOSED at 01:10 MSK (no new training after 02:00 anyway). V2 19b05dad = best promoted.
- All results: research/native_b_sampler/NIGHT_SUMMARY.md + RESULT_*.md; #43 final status comment; assets
  721613794 (V2 + logs), 722010850 (oracle evidence for #47), 722097549 (research checkpoints, not promoted).

## Done via #51
- R19-TPHASE-S0 (Grigoriy card #51/6714199745): DONE 02:10, control .61482/.55916, tphase .61932/.56291, dP +.0045
  dB +.0087 INCONCLUSIVE; asset 722232793; branch research/sanya-w51-r19@0f8a21f; out C:
doe-out
19_s0_*.
- R19-TPHASE-S1 (card 6714389928/6714461127): DONE 02:38, dP +.0057 dB +.0088 REPLICATED, 2-seed mean +.0051/+.0087
  below .010 gate -> not promoted; asset 722289587; out C:
doe-out
19_s1_*.

## Queue = GitHub #51 (worker, `C:
uclear-doe-operator\ops\worker51\`, copy in `ops_worker51/`)
- Only cards with TASK_ID + TO=@MVRto01/DESKTOP-T1K698V + STATUS=READY from NinjaMoons / ggmaster913 (with
  SCIENCE_GATE=GO or explicit coordinator GO). Procedure: `ops\worker51\WORKER.md`. Ledger: `state.json`.
- No repeats (R16/R17/R18/R8/B75/capacity/distill/recipe/STACK). V2 19b05dad (asset 721714285) stays protected;
  replace only with independent acceptance.

## Time gates (MSK, coordinator #51/6714140160 supersedes 02:00/04:00)
02:00-04:30 bounded preregistered TRAIN-only training with GO · 04:30-05:15 finish/export/compare only ·
05:15-06:00 acceptance/packaging · 06:00 freeze · 07:30-08:00 official submission (coordinator) · 09:30 stop coding.

## Recovery
- INDEPENDENT worker: Windows Task Scheduler `nuclear-doe-4070ti-worker` every 15 min (until 09:40 MSK) ->
  C:
uclear-doe-operator\ops\worker51unner.ps1 (mutex, cooldown, precheck without LLM, headless `claude -p` only
  for a new READY task to DESKTOP-T1K698V). Self-test 03:00 from a new headless process: #51/6714786063 (PASS).
  Works after this terminal closes (needs the user logged in + PC awake: AC sleep/hibernate = never).
- Logs: ops\worker51unner.log, runner-state.json, claude-*.json; ledger state.json; lock tick.lock.
- Old in-session CLI cron 5752bdd8: REMOVED 03:17 after the natural scheduled tick 03:16 (PASS). Single poller.
- Claude quota: UNKNOWN from CLI; on a quota error the runner sets a cooldown (no retry loop).
