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
