"""Acceptance run for the STEM contest candidate: runs the packaged CLI in a CHILD process and writes a receipt.
python -m back_projection_stem.acceptance --checkpoint STEM.pt --targets T1 T2 T3 T4 --output-dir NEW --receipt R.json [--synthetic-test]
Checks: completed manifest, exactly one DOE.png (L, uint8, 128x128), four reconstructed_0k.png (L, 128x128) in z order,
native output pitch lambda*z/(N*p) per plane, checkpoint SHA/hparams in metadata, metrics per plane; records wall time
and the child's peak working set (Windows) / max RSS (POSIX). No training, no optimiser, no TEST."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

LAMBDA, PITCH, Z, N = 5.32e-7, 1e-5, (0.0241, 0.0253, 0.0265, 0.0277), 128
CHILD = r'''
import json, sys, time
t0 = time.perf_counter()
from back_projection_stem.infer_neural_stem import main
rc = main(sys.argv[1:])
wall = time.perf_counter() - t0
peak = None
try:
    import ctypes, ctypes.wintypes as w
    class PMC(ctypes.Structure):
        _fields_ = [("cb", w.DWORD), ("PageFaultCount", w.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    c = PMC(); c.cb = ctypes.sizeof(PMC)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), w.DWORD]
    k32.K32GetProcessMemoryInfo.restype = w.BOOL
    if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise OSError("K32GetProcessMemoryInfo failed")
    peak = int(c.PeakWorkingSetSize)
except Exception:
    try:
        import resource; peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        peak = None
print("ACCEPTANCE_CHILD " + json.dumps({"rc": rc, "wall_s": wall, "peak_bytes": peak}))
'''


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--targets", nargs=4, required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--synthetic-test", action="store_true")
    a = ap.parse_args()
    out = Path(a.output_dir)
    argv = ["--targets", *a.targets, "--output-dir", str(out), "--checkpoint", a.checkpoint] + (["--synthetic-test"] if a.synthetic_test else [])
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-c", CHILD, *argv], capture_output=True, text=True)
    child = next((json.loads(l.split(" ", 1)[1]) for l in proc.stdout.splitlines() if l.startswith("ACCEPTANCE_CHILD ")), None)
    checks, rec = {}, {"argv_cli": ["python", "-m", "back_projection_stem.infer_neural_stem", *argv],
                       "returncode": proc.returncode, "stderr_tail": proc.stderr[-2000:], "child": child,
                       "process_wall_s": time.perf_counter() - t0, "python": platform.python_version(),
                       "platform": platform.platform(), "checkpoint_sha256": sha(a.checkpoint),
                       "inputs": [{"path": t, "sha256": sha(t)} for t in a.targets]}
    if proc.returncode == 0 and (out / "manifest.json").exists():
        man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        met = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        md = man["metadata"]
        does = list(out.glob("DOE*.png"))
        doe = Image.open(out / "DOE.png")
        recs = [Image.open(out / f"reconstructed_0{k}.png") for k in range(1, 5)]
        checks["completed"] = bool(man.get("completed"))
        checks["one_doe_png_L_uint8_128"] = len(does) == 1 and doe.mode == "L" and doe.size == (N, N)
        checks["four_recons_L_128_ordered"] = all(r.mode == "L" and r.size == (N, N) for r in recs) and \
            [p["reconstruction"] for p in met["planes"]] == [f"reconstructed_0{k}.png" for k in range(1, 5)]
        checks["z_order"] = [p["distance_m"] for p in met["planes"]] == list(Z)
        checks["native_pitch"] = all(abs(man["output_pitches_m"][k][0] - LAMBDA * Z[k] / (N * PITCH)) < 1e-15 for k in range(4))
        checks["inputs_archived_in_order"] = [sha(out / "targets" / f"target_0{k}.png") for k in range(1, 5)] == [i["sha256"] for i in rec["inputs"]]
        checks["metadata_stem_12ch"] = md["hparams"].get("stem") == "backprojection_v0" and md["hparams"].get("in_channels") == 12
        checks["metadata_checkpoint_sha"] = md["checkpoint_sha256"] == rec["checkpoint_sha256"]
        checks["cli_method_neural"] = md.get("cli_method") == "neural"
        checks["metrics_primary_from_reloaded_png"] = met["primary_source"] == "reloaded_uint8_PNG_from_reloaded_DOE"
        rec.update({"metrics": [{"z_m": p["distance_m"], "pearson": p["pearson"], "ssim": p["ssim"]} for p in met["planes"]],
                    "timing": {k: md.get(k) for k in ("checkpoint_load_elapsed_s", "inference_elapsed_s")} | {"export_s": man.get("elapsed_export_s")},
                    "git_sha": man.get("git_sha"),
                    "outputs": {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}})
    rec["checks"] = checks
    rec["PASS"] = bool(checks) and all(checks.values())
    Path(a.receipt).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps({"PASS": rec["PASS"], "checks": checks, "metrics": rec.get("metrics"), "child": child}, indent=1))
    return 0 if rec["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
