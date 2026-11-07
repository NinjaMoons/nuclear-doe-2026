# HOST_CAPACITY — Matvey home PC (`DESKTOP-0GHPCI9`)

Checked 2026-11-08 01:20 MSK by Claude Code on the host itself (@playerr456). All values were measured, not assumed.

## Hardware / OS
| | |
|---|---|
| HOST_ID | `DESKTOP-0GHPCI9` |
| OS | Windows 11 Pro 10.0.26200 |
| CPU | AMD Ryzen 7 7800X3D, 8C/16T |
| RAM | 31.1 GiB total, ~21 GiB free |
| Disk | C: 95.7 GiB free |
| GPU | **NVIDIA GeForce RTX 5060**, 8151 MiB (≈7.0 GiB free at idle; desktop/apps use ~0.9 GiB), compute capability 12.0 (Blackwell, sm_120) |
| GPU_UUID | `GPU-6e7aa223-755f-477d-fcb3-8dd6fbcd6b77` |
| Driver | 610.88 (CUDA UMD 13.3), WDDM, display attached |
| iGPU | AMD Radeon Graphics (not used) |
| Python jobs at check time | none; GPU util 1–5 % (desktop only) |

## Software
| | |
|---|---|
| Python | 3.12.16 (uv-managed standalone), repo `.venv` |
| PyTorch | **2.9.0+cu128** (same version as `requirements-torch-cpu.txt`, CUDA wheel from download.pytorch.org/whl/cu128), cuDNN 9.10.2, arch list includes sm_120 |
| numpy / pytest / pillow | 2.3.4 / 8.4.2 / 13.0.0 |
| Git / gh / uv | 2.56.0 / 2.103.3 (portable, `~/tools/gh`) / 0.13.7 (`~/tools/uv`) |
| gh auth | `playerr456`, scopes repo, read:org, gist |
| Claude Code | desktop app (no `claude` CLI on PATH) |
| CUDA Toolkit / nvcc | not installed (not needed for PyTorch wheels) |

Note: RTX 50-series needs a CUDA ≥12.8 build of PyTorch. CPU wheels or cu121/cu124 builds will not run on this GPU.

## GPU smoke test — PASS
Script `experiments/smoke/gpu_smoke.py` (local, not tracked). It uses synthetic data only, so no training was done.

| check | result |
|---|---|
| `torch.cuda.is_available()` / device_count | True / 1 |
| fp32 matmul 1024² vs CPU, max rel err | 5.3e-7 |
| matmul throughput 4096² | fp32 11.2 TFLOPS, bf16 34.6 TFLOPS |
| complex64 FFT2 round-trip 8×512² | max err 1.8e-6 |
| CNN fwd/bwd, 50 Adam steps (25 fp32 + 25 AMP bf16), batch 8×4×256² | 0.79 s, all losses finite, loss decreased, peak 356 MiB |
| 5 GiB VRAM allocation | OK |
| repo `pytest -q` | 8 passed |

**Rough capacity vs Sanya RTX 4070 Ti 12 GB.** fp32 throughput is about 0.3–0.5×. VRAM is 8 GB, so W32/24k-class runs may need a smaller batch or AMP. Use this host for small pilots, independent replays and benchmarks, not as a 1:1 replacement.

## Local layout (outside Git tracking)
- Repo: `C:\Users\rybka\work\nuclear-doe-2026`, work branch `research/matvey-homepc-5060`
- `experiments/`, `logs/` and `checkpoints/` are excluded via `.git/info/exclude`. `checkpoints/` and `*.pt` are also in `.gitignore`.

## Remote access — NOT CONFIGURED
- OpenSSH **client** present. OpenSSH **server** not installed. RDP disabled (`fDenyTSConnections=1`). No Tailscale/ZeroTier. Both network profiles (Ethernet, VPN `tun0`) are **Public**.
- Nothing was opened or changed. Enabling sshd/RDP is a system security change for the PC owner to make.
- Working options without opening ports:
  1. **Option B (recommended tonight):** an independent Claude Code session on this PC, with coordination via GitHub Issues and the branch. The Claude desktop app session can also be followed and steered from another device.
  2. If SSH is needed: the owner installs OpenSSH Server, uses key-only auth, and restricts the firewall to the trusted LAN or a private overlay (e.g. Tailscale). No public port forwarding.

## Quota
Claude 5h/weekly usage is not visible from this session. Read it from the account UI. It is shared with the laptop session.
