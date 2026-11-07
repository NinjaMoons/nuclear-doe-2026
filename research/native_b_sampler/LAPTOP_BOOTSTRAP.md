# Sanya laptop = HOST2 — bootstrap (mode B: own Claude Code session, GitHub locks). Status: NOT VERIFIED.

Why mode B: from the desktop (DESKTOP-T1K698V, 192.168.1.101) no LAN host answers on TCP 22 and the desktop has no SSH
keys/config, so mode A (desktop Claude -> SSH worker) needs the laptop owner to enable Windows OpenSSH Server + a key
on the trusted LAN first. Do not open SSH to the Internet.

## 1. Probe (PowerShell on the laptop, paste the output into #51 with secrets removed)
```powershell
hostname; Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, AdapterRAM
nvidia-smi -L; nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.free,utilization.gpu --format=csv,noheader
Get-PSDrive -PSProvider FileSystem | Select-Object Name, Free
Get-Process python, pythonw -ErrorAction SilentlyContinue | Select-Object Id, StartTime
py -0p; claude --version; gh auth status; ssh -V
```
No NVIDIA GPU / no CUDA -> HOST2 = CPU worker only (QA, sheets, docs). Never call an iGPU/AMD GPU "CUDA".

## 2. Environment + 100-step CUDA smoke (only if nvidia-smi works)
```powershell
gh repo clone NinjaMoons/nuclear-doe-2026 C:\ndoe-laptop; cd C:\ndoe-laptop
git fetch origin; git worktree add -b research/sanya-laptop-<experiment> ..\ndoe-laptop-wt origin/research/sanya-native-b-sampler
py -3.11 -m venv C:\ndoe-venv; C:\ndoe-venv\Scripts\python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
C:\ndoe-venv\Scripts\python -m pip install numpy==2.3.4 Pillow==13.0.0
C:\ndoe-venv\Scripts\python -c "import torch;print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))"
```
The native corpus (`corpus-4b`, manifest 76d2474d) comes from the private draft `data-4b-v1`; delete the TEST sample
dirs after extraction (TEST must never be decoded). Smoke: `python research/wc_stem_pilot/pilot.py train --corpus <corpus>
--arm native --seed 0 --updates 100 --ckpt-every 100 --out C:\ndoe-out\smoke` -> report s/step and peak memory.

## 3. Claiming work
- Before any GPU job: comment in #51 `CLAIM EXPERIMENT_ID=<id> HOST=<hostname> GPU_UUID=<uuid> BRANCH=<branch> OUT=<dir>`.
- Desktop owns #43 queue (RECIPE W32d3 24k, then R17-BRECOMB). Laptop must pick a DIFFERENT IDEA_CARD from #49, or do
  independent CPU/GPU replay of desktop candidates (`research/desktop41_p0/replay_stem_val.py`) and #47 raw sheets.
- Own output dirs only (`C:\ndoe-out\...`), never the desktop's; no shared checkpoints; one GPU job at a time.
- Same Claude subscription quota is shared with the desktop session if the same account is used.

## 4. Startup prompt for the laptop Claude Code
"You are the laptop Claude Code of MVRto01 (HOST2). Read CLAUDE.md, origin/main docs/hackathon/NIGHT_AGENT_PROTOCOL.md
and DUAL_HOST_SETUP.md, issues #51, #49, #43. Run the probe in research/native_b_sampler/LAPTOP_BOOTSTRAP.md, post the
real result to #51, and only then claim one non-overlapping task. Never touch the desktop's dirs or jobs."
