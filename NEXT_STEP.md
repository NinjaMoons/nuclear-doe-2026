# NEXT_STEP — @MVRto01 desktop Claude (#43 / #51) — updated 2026-11-08 00:10 MSK

Read this first after any quota reset / restart. Then `git fetch origin` and re-read #43, #49, #51, #8 and
`git show origin/main:docs/hackathon/NIGHT_AGENT_PROTOCOL.md`. Never restart a job whose outputs already exist.

## Hosts
- HOST1 = DESKTOP-T1K698V, RTX 4070 Ti 12 GB, GPU-282fb2b9-94df-8922-4941-d3b30604ee5f, driver 616.56,
  venv `C:\nuclear-doe-operator\desktop-41\.venv-41` (py3.11.9, torch 2.9.0+cu128). ONE GPU job at a time.
- HOST2 = Sanya laptop: **UNVERIFIED** (no hostname/GPU known; no SSH server reachable on LAN 192.168.1.0/24, desktop
  has no SSH keys). Mode B (separate Claude Code on the laptop + GitHub locks) unless the owner enables OpenSSH there.
  Bootstrap for it: `research/native_b_sampler/LAPTOP_BOOTSTRAP.md`.

## Running / finished on HOST1 (check with `nvidia-smi --query-compute-apps=pid,process_name --format=csv`)
- RECIPE (pre-registered `RESULT_DISTILL.md`): W32d3 24000 updates, const vs cosine lr, seeds 0/1.
  Script `C:\nuclear-doe-operator\desktop-41\recipe\run.sh`, log `...\recipe\run.log`, out dirs
  `...\recipe\W32_{cosine,const}_s{0,1}`, eval `...\recipe\eval\eval.json` (written at the end, then `ALL_DONE`).
  Started 23:35; PID at 00:00 = 36824 (seed1 cosine). If `ALL_DONE` is in the log, analyse; do NOT rerun.
- DONE earlier: WC pilot (WC_NEUTRAL), long native (V2 19b05dad), B75 (NEUTRAL), oracle, capacity (NEUTRAL),
  distill (NEUTRAL). Evidence: #41/6711603518, #43/6712097137, /6712581459, /6712712710.

## Queue (decide GO/NO-GO yourself; post evidence to #43)
1. Analyse RECIPE vs V2 (P .61243 / SSIM .55356 / B .4516 / B own 25/32). Gate: dP >= +.01 vs V2, dB >= +.005,
   orgsets10 & WC64 not worse -> CPU CLI replay (`research/desktop41_p0/replay_stem_val.py`) + #40-runtime drop-in
   (`desktop-41\ref40\pkg\...\src`, acceptance) + raw sheets A64:060/B64:012/B64:016, then upload private asset.
2. **R17-BRECOMB** (IDEA_CARD f9239c5, routed to MVRto01): code ready `research/native_b_sampler/r17_recomb.py`
   (pool 118 verified, leak guard OK). Arms control/recomb x seeds 0/1, 6000 updates, init V2. MUST START BEFORE
   02:00 MSK. Then eval with `research/wc_stem_pilot/pilot.py eval ... --evalsets C:/nuclear-doe-operator/desktop-41/evalsets`.
3. If nothing promotable: P0 support (#8/#47 raw sheets, cross-host replay of V2), no new training after 02:00.

## Time gates (MSK)
02:00 no new training · 04:00 research stop · 05:30 freeze · 07:30-08:00 official submission (designated submitter only).

## Recovery
- `opsesume43.ps1` (status log without AI every run + ONE headless Claude resume, cap 3, >= 50 min apart, only if
  no interactive Claude CLI is alive) is written and DRY-RUN TESTED (00:01: detected PID 36824 job, skipped because
  the interactive session was alive). The Windows task is **NOT registered** (registration from the agent was
  denied by the permission classifier). Until the owner registers it: **RESUME_MANUALLY** = `claude --continue`
  in `C:
uclear-doe-operator-sampler`. Live headless launch has never been tested on this host.
- Claude quota: not visible from this CLI session (UNKNOWN). Last user-reported: session 34% USED, reset 00:50 MSK.
