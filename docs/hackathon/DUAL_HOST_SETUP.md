# Two independent GPU computers / one or two Claude Code sessions — deployment guide

**Owner:** @MVRto01 (Sanya). **Status:** SETUP REQUIRED, GPU 2 NOT VERIFIED; no SSH or agent process established by writing this file. Read [NIGHT_AGENT_PROTOCOL](NIGHT_AGENT_PROTOCOL.md) and #43 first.

## Architecture choices — recommend A by default

**A. One Claude Code coordinator on Sanya desktop + remote GPU worker on laptop (preferred if secure SSH exists).** Desktop Claude plans/schedules, uses normal local commands for desktop GPU and SSH remote shell for laptop GPU, with each training job started as a persistent independent OS process. A single Claude session can orchestrate both **after** laptop host/SSH is configured. One shared Claude account quota is used; remote GPU runs do not themselves use Claude inference tokens. Persist local logs/checkpoints and GitHub evidence.

**B. One Claude Code session on each computer.** Both read same repo docs/Issues and post host/job reservations, but may share the SAME Pro plan quota if authenticated to same account. Two clients do not create twice the subscription limit. Separate Git worktrees, local env and output dirs. This is more fault tolerant if desktop orchestrator stops, but requires coordination and account-auth legitimacy. **Don't install second session merely to use GPU**, as a GPU worker needs Python/Torch/CUDA and remote control, not its own Claude login.

Choose A if SSH works over local trusted LAN/VPN; choose B if each computer is independently operated and authorized. Never expose SSH to the public internet simply to meet deadline; don't publish passwords/tokens or tunnel to unknown hosts.

## HOST PROBE — run locally on EACH Windows computer (PowerShell)

Run inside relevant repo folder in each host, save results with secrets stripped. Do not assume NVIDIA exists on notebook until verified:

```powershell
hostname
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
Get-PSDrive -PSProvider FileSystem | Select-Object Name,Free
Get-Process -Name python,pythonw -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime
Get-Command nvidia-smi -ErrorAction SilentlyContinue
nvidia-smi -L
nvidia-smi --query-gpu=name,uuid,memory.total,memory.free,utilization.gpu --format=csv,noheader
python -c "import torch; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'count',torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
git status --short
git branch --show-current
git rev-parse HEAD
gh auth status
claude --version
ssh -V
```

`nvidia-smi`, `gh`, `claude`, `ssh` may be missing; report `NOT_INSTALLED` rather than falsify. GPU must be independently validated with CUDA torch in an **isolated venv**. If notebook GPU is AMD or iGPU, do not claim it is CUDA.

**Report template in #43**:
`HOST_ID=...; IP/SSH_REACHABLE=YES/NO (omit public IP); GPU_NAME/UUID=...; VRAM_FREE=...; TORCH_CUDA=...; JOB_PIDS=...; FREE_DISK=...; REPO_COMMIT=...; QUOTA_USED_REMAINING/RESET_TZ=...; TASK_LOCK=...`

## GitHub setup: two machines, no shared uncommitted state

1. Existing user signs into their own GitHub on each computer (check `gh auth status`); do not share login tokens in chat.
2. Use a separate local clone or worktree per independent experiment. If repo not cloned, after authenticating:
```powershell
gh repo clone NinjaMoons/nuclear-doe-2026
cd nuclear-doe-2026
git fetch origin
git status --short
```
3. Read `CLAUDE.md`, `docs/hackathon/NIGHT_AGENT_PROTOCOL.md`, Issue #43 and #49. Create a unique branch `research/sanya-laptop-<hypothesis>` for laptop; desktop branch remains untouched. Don't create/swap refs while running process writes code/checkpoint.
4. Assign one `EXPERIMENT_ID` per distinct hypothesis; mark `CLAIMED/ACTIVE` with `HOST_ID + GPU_UUID + dataset + branch` in GitHub before launching, and `DONE/NO_GO + SHA` afterward. If desktop and laptop propose same model/teacher labels, assign **one runner only**, the other does independent metrics/physics/defense QA.
5. Use immutable references for input data, separate local output dirs and logs. Never overwrite v1/v2/4ch official candidate files. Don't push TEST or restricted dataset to Git.

## Trusted SSH setup — only if explicitly enabled by the laptop owner

- For A, on notebook configure Windows OpenSSH Server through Windows Settings > Optional Features or the approved IT process; use key authentication scoped to a limited user, trusted LAN/VPN only, Windows firewall restricted to trusted subnet. **Do not enable public Internet SSH port forwarding or share credentials.**
- On desktop use `ssh -V`, then test with owner-provided local host alias:
```powershell
ssh gpu-notebook "hostname"
ssh gpu-notebook "nvidia-smi -L"
```
- If SSH doesn't work, don't waste deadline time debugging a network stack; choose B (separate local Claude session) or use the notebook locally to run a pre-packaged Python experiment and publish its results to GitHub.
- Remote job orchestration must quote paths and use the remote user's venv. Before a long run, do a **100-step smoke**, verify remote stdout log, GPU PID, and that job survives closing the initiating terminal. Windows process survival depends on how the job is started: use an owner-authorized Scheduled Task/service for an independent background job if necessary; do not assume `Start-Process` always detaches safely.
- No simultaneous jobs per GPU. No hidden password manager export or privilege elevation.

## Quota and reset: don't equate GPU compute with Claude chat

- Every Claude client reads its own actual account UI; record **USED vs REMAINING**, screenshot-local timezone, 5h reset, weekly reset, checked timestamp. A shared subscription's quota may apply across both clients; neither second host nor a new CLI invocation grants additional quota.
- When Claude API quota is temporarily exhausted, training already launched under a trusted OS supervisor can continue. A deterministic local watcher may check PIDs, log files and metrics **without AI calls** and append `RUNNING/FINISHED/FAIL` to a local status file. If GitHub update credentials are available, rate-limit publishing; never spam requests.
- For CONTINUATION, the agent writes `NEXT_STEP.md`: command/host/PID, Git commit, current checkpoint SHA, all completed tasks, pending issue, next allowed action, quota error and exact reset time (TZ), safe re-entry steps. The person or a pre-authorized, tested scheduler can run `claude --continue` / `claude --resume` **after** reset. Neither of those commands self-starts after reset. Unattended noninteractive retries of Claude Code require separate installation, authorization and failure tests; do NOT assert installed.
- Never create a tight auto-retry loop around 429/quota errors; don't attempt to bypass provider usage policy.
- If the experiment ends while CLI is rate-limited, deterministic code can save artifacts; independent human/agent pickup on resume compares actual files before deciding whether to rerun. Never rerun completed GPU training solely because the chat has no response.

## One-off task assignment once laptop CUDA confirmed

**Desktop:** finish already ACTIVE train-only teacher/student distillation (#43). Its current loss uses direct circular phase alignment; Matvey's `R16 optical_display` is a related alternative, NOT a duplicate full training run without research justification. No premature renaming of done/candidate.

**Laptop:** ask Grigoriy #49 for the highest-ranked **non-overlapping** tested research card; potentially B quartet recombination after true independence/provenance tests, frequency-aware loss pilot with anti-speckle controls, or train-only DATA generator validation, each one-factor and pre-registered. Use exact same official output metric pipeline and compare with V2 (P=.61243 SSIM=.55356 B=.4516, 25/32). If time is too short, use its GPU for independent fast replay/benchmark of desktop's candidate rather than speculative training.

**Real stop times:** New training branches only before 02:00 MSK and if p90 complete+review before 04:00; release evidence and slides freeze 05:30; aim official submission 07:30–08:00.

## Confirmation checklist

- [ ] Both machines report HOST_ID and actual GPU model/VRAM/torch CUDA, no mystery process.
- [ ] Both machines read and ACK root `CLAUDE.md` and protocol after `git fetch`.
- [ ] One unique EXPERIMENT_ID / one physical GPU job each.
- [ ] Remote SSH key/trusted network tested IF A chosen, or independent separate local session if B chosen.
- [ ] GPU jobs save to durable local checkpoints and can survive a terminal disconnection, tested with smoke.
- [ ] Claude quota used/remaining + reset + timezone and safe resume mechanism documented.
- [ ] GitHub latest tasks and protected candidate SHAs verified; no TEST/whitecollars violations.
- [ ] At least one independent V2 P0 acceptance and official submission/defense owner remains allocated.


## Supported scheduling choices (official Claude Code docs)

In addition to OS Task Scheduler, Claude Code supports:
1. `/loop 10m ...` inside an *open* CLI for short polling while the session runs. It does **not** survive exiting without resume and consumes quota. Avoid tighter intervals.
2. **Claude Desktop local scheduled tasks**: local GPU/files accessible, new session starts on schedule even if CLI session closed; the Desktop app must be open and the PC awake. This is best for a **single scheduled recovery check after a known quota reset** if installed. Runbook must read NEXT_STEP and detect already-running job first.
3. **Claude Code cloud `/schedule`**: can run when PC off and read GitHub, but no access to local GPU in standard Anthropic cloud environment. Rejected when subscription limit exhausted (unless authorized overage); do not use it to bypass quota.
Both require setup in the user's Claude account and verification in Scheduled tasks UI; this repo instruction does NOT create them. Only designated owner enables any paid/overage setting, never agent without approval.
Docs: https://code.claude.com/docs/en/desktop-scheduled-tasks , https://code.claude.com/docs/en/routines , https://code.claude.com/docs/en/scheduled-tasks .
