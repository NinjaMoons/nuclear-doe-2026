"""RESEARCH ONLY: pack research runs into a compact ZIP; hashes of every file (packed or LOCAL_ONLY) are listed.

Usage: python -I pack_runs.py <research_root> <out_zip>
Packed: everything except eval_val/**/raw.npz and periodic checkpoints/step_*.pt (listed as LOCAL_ONLY).
"""
import hashlib
import sys
import zipfile
from pathlib import Path

root, out = Path(sys.argv[1]), Path(sys.argv[2])
assert not out.exists()
packed, local_only = [], []
for base in ("runs", "reports"):
    for p in sorted((root / base).rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        skip = (p.name == "raw.npz" and "/eval_val/" in rel) or (p.parent.name == "checkpoints" and p.name.startswith("step_"))
        (local_only if skip else packed).append((h, rel, p))
sums = "".join(f"{h}  {rel}\n" for h, rel, _ in packed)
lo = "".join(f"{h}  {rel}\n" for h, rel, _ in local_only)
with zipfile.ZipFile(out, "x", zipfile.ZIP_DEFLATED) as z:
    for _, rel, p in packed:
        z.write(p, rel)
    z.writestr("SHA256SUMS.txt", sums)
    z.writestr("LOCAL_ONLY_SHA256SUMS.txt", lo)
    with zipfile.ZipFile(out.with_suffix(".tmp"), "w") as _:
        pass
out.with_suffix(".tmp").unlink()
print(len(packed), "packed;", len(local_only), "local-only;", out.stat().st_size, "bytes;",
      hashlib.sha256(out.read_bytes()).hexdigest())
