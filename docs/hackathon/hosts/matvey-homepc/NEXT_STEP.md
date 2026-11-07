# NEXT_STEP — Matvey home PC (`DESKTOP-0GHPCI9`, RTX 5060 8 GB)

Status 2026-11-08 01:25 MSK: **READY_FOR_TASK**. No job is running and the GPU is idle. No experiment has been claimed.

## Re-entry (any new Claude Code session on this PC)
```powershell
cd C:\Users\rybka\work\nuclear-doe-2026
$env:Path = "C:\Program Files\Git\cmd;$env:USERPROFILE\tools\gh\bin;$env:USERPROFILE\tools\uv;" + $env:Path
git fetch origin; git status -sb
nvidia-smi
Get-Process python* -ErrorAction SilentlyContinue
.venv\Scripts\python.exe -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Then re-read `CLAUDE.md`, `docs/hackathon/NIGHT_AGENT_PROTOCOL.md` and the latest comments of #50, #49 and #43.

## Next allowed actions (pick ONE after checking R16/R17 owners' results)
1. Wait for an assigned task in #50 or #49. Do not duplicate R16 (done, INCONCLUSIVE), R17, R8 or R18.
2. Best fit for this host:
   - an independent GPU replay or acceptance of a candidate another host produced, on the official pipeline (DOE u8 → reload → 4 PNG → P/SSIM);
   - a short one-factor pilot that fits 8 GB.
3. Before any run:
   - claim `EXPERIMENT_ID` + `HOST_ID` + `GPU_UUID` + branch in the Issue;
   - do a 100-step smoke run;
   - start the job as an OS process that survives the terminal;
   - write logs to `logs/` and checkpoints to `checkpoints/`.

## Hard gates (MSK)
- No new training hypothesis after 02:00.
- Research stops at 04:00.
- Freeze at 05:30.
- Never touch V2 `721714285`, V1 `721351745` or 4ch `720489303`, and never touch TEST or WhiteCollars.

## Recovery
`RECOVERY_MECHANISM=MANUAL`: no scheduler has been installed or tested. Remote access is via the Claude app and GitHub only (no SSH/RDP).

## HOME_HANDOFF snapshot — 2026-11-08 01:30 MSK (Claude quota ~91% USED, shared)
- Workdir: `C:\Users\rybka\work\nuclear-doe-2026`
- Branch: `research/matvey-homepc-5060`. The base for this handoff was `ca5d997`; the handoff commit itself is the branch tip, see `git log -1`.
- Working tree: clean. Only ignored local dirs exist (`.venv/`, `experiments/`, `logs/`).
- CUDA venv: `.venv\Scripts\python.exe` (Python 3.12.16, torch 2.9.0+cu128, numpy 2.3.4, pytest 8.4.2, pillow 13.0.0)
- GPU: RTX 5060 `GPU-6e7aa223-755f-477d-fcb3-8dd6fbcd6b77`, 855 MiB used (desktop apps only), util ~8 %
- Python PIDs: **none**. No training or replay job is running and no experiment has been claimed.
- Logs: `logs\gpu_smoke_20261108.json` (smoke PASS). Smoke script: `experiments\smoke\gpu_smoke.py`.
- Scheduled tasks: none for Claude/recovery. The only matches were Windows built-ins (`Recovery-Check` etc.), so recovery stays **MANUAL**.

### Re-entry command (after quota reset; run by a human)
Open the Claude desktop app → Code → folder `C:\Users\rybka\work\nuclear-doe-2026`, then give this prompt:
"Read docs/hackathon/hosts/matvey-homepc/NEXT_STEP.md, run the re-entry block, read the latest #50/#49, report READY and take only an assigned task."
For a shell-only check, run the PowerShell block in "Re-entry" above.

### Available P0 tasks (none started here)
- **V2 `721714285` replay acceptance** on CUDA, read-only. Procedure: `docs/hackathon/defense/REPRO_CHECKLIST.md` on branch `origin/ops/repro-checklist-playerr456`.
  - Not run in this handoff: it needs a private release-asset download and data setup.
  - #50 A was already DONE on the laptop.
  - Run it only if the coordinator asks for cross-host GPU confirmation.
- Independent GPU replay of any new candidate posted by R17/R8/R18 owners, on the official pipeline (DOE u8 → reload → 4 PNG → 4×P/SSIM).
- Defense QA from #47/#50 B (manifest/metrics cross-check) — CPU.
