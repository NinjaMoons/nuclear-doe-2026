# Nuclear IT Hack · NIGHT_AGENT_PROTOCOL · 7–8 November 2026

> CURRENT directive from Pavel as coordinator, valid for final night. **This is a durable per-repository Claude Code work rule**, not a running daemon. Every local or remote Claude session must `git pull` and re-read `CLAUDE.md`, `AGENTS.md`, own Issues and this document. Respect existing safety/contract boundaries; newer coordinator-approved tasks supersede older HOLD only where explicitly stated.

## Mission and source of truth

Deliver the strongest eligible **neural** multi-plane DOE model before deadline, reproducible code/model, four raw amplitude recon PNG, per-z Pearson and global SSIM, Git-hosted defense source and a REAL official submission receipt. Current protected native-only STEM V2 `721714285`, VAL16 P=.61243 SSIM=.55356 B_P=.4516, B own 25/32; V1 `721351745` and 4ch backup `720489303` immutable. Group B ghosting still visible. Training-only per-sample Adam teacher is permitted for TRAIN label creation subject to contest rules; per-input Adam/GS at contest inference is NOT allowed. Frozen TEST never used for selection. No WhiteCollars-trained weights without written rights and domain audit.

**The central engine is two Gemini research streams + Grigoriy's Claude Code as scientific verifier/dispatcher, with operational owner/coordinator @NinjaMoons.** Never treat a raw Gemini answer as a fact. Gemini cannot auto-write GitHub unless specifically connected: raw prompt + full output must be transferred intact to Grigoriy and committed to Git, scientific checks first. See [#49](https://github.com/NinjaMoons/nuclear-doe-2026/issues/49).

## Mandatory agent loop (when actually running)

1. Sync current repo/Issue/comments and check `HOST, GPU, python jobs, dependency, deadline, quota/reset`. Do not assume missing GitHub comments mean process idle. `claude` CLI sessions do not watch new issues by themselves.
2. Select one highest-value **non-conflicting** ready task. Track ownership per *physical host + distinct scientific factor*. Pre-register each training pilot and define exact TRAIN split, matched control, VAL test, B raw 4-plane visual samples, stop rules and end-to-end p50/p90.
3. Execute in owned branch/worktree without overwriting protected checkpoints, releases or another owner's code. Review scientific hypothesis first if risky, then health+paired pilot, not blind parameter search. Save log and checkpoint periodically on local disk (not git), push source/manifest/small metrics and protected release with SHA after verification. Never check secrets or giant binary dataset into git.
4. On DONE/NO_GO/BLOCKED, post `STATUS | OWNER | HOST/GPU/JOB_PID | SOURCE_SHA | INPUT_SPLIT | MODEL_SHA | METRICS_A_B_WORST_SS | B_OWN/4X4 | 4Z_RAW_ASSET_SHA | QUOTA_USED_OR_REMAINING (state which) | RESET_TIME + TZ | P50/P90 | NEXT_ACTIVE | BLOCKER | SUBMISSION_RECEIPT`; immediately select NEXT permissible task. If no viable research, switch to P0 test, packaging, assets/defense or official submission procedure. Idle waiting for a routine Pavel confirmation is not needed.
5. GitHub is the coordination interface; add links to actual current SHA and files, not just chat messages or screenshots. External Gemini/Claude outputs must become Git files/Issue cards. Competing results require matched identical quartets, official 8-bit DOE round-trip + 4 amplitude PNG reload and independent cross-host review.
6. An agent may make bounded engineering decisions autonomously (train, compare, select, prepare separate candidate ZIP and defense, request peer review), but not violate rules/rights, spend money without approval, rewrite protected code, or perform **final irreversible official contest upload** unless designated submitter explicitly authorized by Pavel. Escalate only material blocker and keep working independently.

## Host-aware work split (DYNAMIC; not permanent)

- **Sanya desktop RTX4070Ti**: #43 GPU ACTIVE currently train-only teacher→student distillation after WIDTH/DEPTH CAPACITY_NEUTRAL. **Do not interrupt or duplicate teacher experiment.** Current student objective is circular phase alignment, which is scientifically distinct from Matvey's proposed optical-display R16; Grigoriy should inspect both loss choices and label integrity, not prematurely assert parity.
- **Sanya laptop**: user reports second discrete GPU, **MODEL/VRAM/CUDA/JOBS UNVERIFIED**. New Claude Code session may work independently with own repo clone/worktree, actual GPU host ID and status, and selected task from #49/Gemini not already running on desktop. If CUDA unavailable, CPU test/visual/data prep. Never let both sessions write shared weights/output dirs. Sessions may share the same Claude subscription quota even with two physical GPUs. One orchestration Claude on desktop can SSH into laptop and manage distinct jobs, but commands only run if network/auth/remote environment are configured; better two active sessions + GitHub coordination for failure isolation.
- **Matvey notebook i5-13420H 16 GB Intel UHD, torch CPU-only**: no expensive GPU train. #50 P0 clean-host release smoke, cross-platform docs, metadata/source/image manifests, help editable defense; R16 optical distillation **HOLD: overlapping train-only teacher idea with Sanya #43**. Can review Sanya distill loss/label alignment instead. Matvey *home PC through brother* unverified: audit exact GPU CUDA VRAM before independent training.
- **Grigoriy Mac M1 + two Gemini**: scientific gateway #49, full raw Gemini output archival, read-only physics / stats / code review of active student, independent model audit and #47 honest B failure narrative, free NEXT = verify other models or defense claims.
- **Coordinator @NinjaMoons**: release gate #8 and #47 defense source, resolve resource collisions, official STEM allowance and form/receipt, collect evidence. Can assign next tasks independent of fixed roles. **No last-minute P0 starvation**.

## Conflict detection and physical work separation

- Report `HOST_ID, GPU_UUID (if available), torch.cuda.device_count, VRAM, gpu utilization, PID/jobs, working tree, checkpoint dirs, branch, disk, network connectivity` before any new job on a host. Run `nvidia-smi` only where installed, inspect from Python otherwise. A laptop GPU and desktop GPU are independent only when confirmed; treat shared disks and credentials separately.
- Issue `EXPERIMENT_ID` and claim `ACTIVE` **before** launching. If same underlying hypothesis exists on another host (for example R16 teacher student vs #43 distillation), stop duplicate; pivot to alternative one-factor method (e.g. B quartet recombination ONLY after proof of valid independent grouping and train-only no leakage, novel spectral read-only experiments, or P0).
- No concurrent expensive GPU process per device. TRAIN512 only until suitable authorized new data. For **new generated data**, ask provenance, creator permission/license, exact 4-plane quartet mapping, amplitude semantics, source distribution matching and TRAIN/VAL/TEST separation before GPU. An offer of 16k unlabeled images is not automatically usable training data.
- Avoid scattershot sweeps on 16 VAL quartets. Never call a no-go result a hidden-test success.

## Claude usage accounting and session recovery

- Each agent obtains usage from **their own visible account UI**, explicitly says whether percentage is USED or REMAINING, exact 5h reset time **with timezone**, weekly number and reset, snapshot time. CLI may not expose quota. Do not assume % of a rolling limit = transferable tokens or infer a precise refill from the word '5-hour'. P50/P90 must include limit reset risk, not only GPU compute.
- Screenshot from Pavel ~23:3x MSK: Grigoriy Pro 5-hour **81% used** (orange bar), text `Resets at 11:50 PM`; weekly **22% used** (`Resets Sat 9:00 AM`). Local timezone of screenshot needs confirmation before converting to MSK; reset is imminent if it matches Moscow. Sanya previously current-session 34% used and reset 00:50 MSK; weekly 16% used; Matvey earlier 5-hour text 43% and reset 02:30 MSK (direction of remaining vs used uncertain).
- If quota near exhaustion: write `NEXT_STEP.md` with exact running PID, branch, task, evidence, time gates, pending work and recovery commands. Training started as an OS job may continue without Claude; don't kill it merely because AI chat quota depleted. Log status via deterministic scripts if configured. At reset, user or explicit trusted scheduler resumes Claude Code, reads `NEXT_STEP.md`, checks already-completed jobs, and takes next item. **A single Claude Code conversation cannot promise to wake up itself after quota reset, nor can a GitHub comment start another Claude instance.** Do not bypass subscription limits or create retry-spam. If scheduler not installed, report `RESUME_MANUALLY` and make continuation trivial. Don't expose subscription credentials or use multiple accounts to bypass terms.
- Avoid using all inference quota right before package freeze; preserve one full final P0 verification / correction pass.

## HARD time gates — MSK

- 02:00 8 Nov: no start of new scientific training hypotheses. Continue existing only with measured p90 completing before 04:00 and independent review capacity.
- 04:00: research stop, select/review candidate, no optimizer branch remaining.
- 05:30: final checkpoint, SHA, figures and presentation source freeze.
- Preferred 07:30–08:00: actual official submission with real receipt; internal team cutoff 09:30, organizer letter absolute by 10:00, Stop Coding 09:00–09:30. Do not count private GitHub release as submission.
- On rule ambiguity or unavailable upload route, P0 escalate early. Don't gamble official compliance for marginal .003 Pearson.

## Recoverability test (everyone)

After pulling this protocol: report one heartbeat in owned issue (not a made-up confirmation):
`ACK_PROTOCOL; ROLE; HOST1/2; CUDA+VRAM; ACTUAL_JOB; LATEST_SHA; QUOTA/RESET/TZ; NEXT_ACTIVE; RECOVERY_MECHANISM=MANUAL_OR_SCHEDULER`.
All prompts and AI conversations are guidance, not a replacement for OS-level independent jobs. Persistent unattended scheduling, if chosen, requires actual installed scheduled job and authorized credentials; do not assert it exists without testing.
