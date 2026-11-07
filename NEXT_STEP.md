# NEXT_STEP — @MVRto01 desktop Claude (#43 / #51) — updated 2026-11-08 00:47 MSK

Read this first after any quota reset / restart. Then `git fetch origin` and re-read #43, #49, #51, #8 and
`git show origin/main:docs/hackathon/NIGHT_AGENT_PROTOCOL.md`. Never restart a job whose outputs already exist.

## Hosts
- HOST1 = DESKTOP-T1K698V, RTX 4070 Ti 12 GB, GPU-282fb2b9-94df-8922-4941-d3b30604ee5f, driver 616.56,
  venv `C:\nuclear-doe-operator\desktop-41\.venv-41` (py3.11.9, torch 2.9.0+cu128). ONE GPU job at a time.
- HOST2 = Sanya laptop: **UNVERIFIED** (no hostname/GPU known; no SSH server reachable on LAN 192.168.1.0/24, desktop
  has no SSH keys). Mode B (separate Claude Code on the laptop + GitHub locks) unless the owner enables OpenSSH there.
  Bootstrap for it: `research/native_b_sampler/LAPTOP_BOOTSTRAP.md`.

## Running / finished on HOST1 (check `nvidia-smi --query-compute-apps=pid,process_name --format=csv`)
- RUNNING: **R8-RES** (PLAN_R8RES.md): `C:
uclear-doe-operator\desktop-418res\phasor_s{0,1}` then eval
  `...8res\eval\eval.json` (log `...8res\eval.log`). Started 00:45. If eval.json exists: analyse vs R17
  control (`...17\eval\eval.json` control_s0/s1 step 6000) with the plan's gate; do NOT rerun.
- DONE (no promotion, V2 stays): WC_NEUTRAL, B75_NEUTRAL, CAPACITY_NEUTRAL, DISTILL_NEUTRAL, RECIPE_NEUTRAL
  (W32 cosine 24k +.005), R17 INCONCLUSIVE. RESULT_*.md in research/native_b_sampler. #43 comments 6712097137,
  6712581459, 6712712710, 6713131129 and the R17/R8-RES status.

## Queue (decide GO/NO-GO yourself; post evidence to #43)
1. Analyse R8-RES. If GO: CPU CLI replay needs the PhasorStemUNet class -> report as research candidate only,
   packaging needs review (#8). If not GO: no new training after 02:00 -> P0 support (#47/#8), independent
   cross-host replay of whatever the coordinator selects, defense numbers from RESULT_*.md.

## Time gates (MSK)
02:00 no new training · 04:00 research stop · 05:30 freeze · 07:30-08:00 official submission (designated submitter only).

## Recovery
- `C:\nuclear-doe-operator\ops\resume43.ps1` (status log without AI every run + ONE headless Claude resume, cap 3,
  >= 50 min apart, only if no interactive Claude CLI is alive) is written and DRY-RUN TESTED (00:01: detected the
  PID 36824 job, skipped because the interactive session was alive). The Windows task is **NOT registered**
  (registration from the agent was denied by the permission classifier). Until the owner registers it:
  **RESUME_MANUALLY** = `claude --continue` in `C:\nuclear-doe-operator\b-sampler`.
  Live headless launch has never been tested on this host.
- Claude quota: not visible from this CLI session (UNKNOWN). Last user-reported: session 34% USED, reset 00:50 MSK.
