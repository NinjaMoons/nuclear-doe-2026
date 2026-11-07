"""R16 teacher-label cache (TRAIN ONLY) + content-addressed lookup for exact label/batch alignment (#49 review, item 1-2).

Labels are keyed by the SHA-1 of the float32 target tensor exactly as the driver holds it (load_png_quartet output), so a
training batch `train_targets[idx]` finds its label by content — no reliance on index order. Each record also stores
the corpus sample key, the 4 target PNG SHA256 and the teacher info. The label is display(|U|) of the teacher phase
after the official uint8 DOE codec (encode -> decode), i.e. in the same space as the submitted DOE (#49 wish (a)).

python -m teacher_student.labels <corpus_dir> <student_ckpt> <out.pt> [--limit N] [--steps 50]
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from back_projection_stem.infer_neural_stem import research_load_checkpoint
from doe_core.contracts_v1 import CaseConfigV1
from doe_core.fresnel import propagate_fresnel
from doe_experiments.data import load_png_quartet
from doe_experiments.phase_codec import decode_phase, encode_phase
from doe_models.adam_multiplane import multiplane_amplitudes
from doe_models.train_multiplane import infer_phase
from teacher_student.losses import display
from teacher_student.teacher import teacher_phase

MANIFEST_SHA = "76d2474d63abcd56759b7ccfe6a780ba6c2ea7968ad8b1d4a688feae8d6f1c78"


def tensor_key(t: torch.Tensor) -> str:
    """Content key of ONE quartet target tensor [4,1,H,W] (float32 contiguous bytes)."""
    return hashlib.sha1(t.detach().to(torch.float32).contiguous().cpu().numpy().tobytes()).hexdigest()


def quartet_tensor(corpus: Path, sample: dict) -> torch.Tensor:
    t = load_png_quartet([corpus / p for p in sample["paths"]])
    return (t if t.ndim == 5 else t[None])[0]                      # [4,1,H,W] float32, same as the driver


def build(corpus: Path, ckpt: str, out: Path, limit=None, steps=50):
    raw = (corpus / "manifest_expanded.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == MANIFEST_SHA, "unexpected manifest"
    man = json.loads(raw)
    samples = {s["key"]: s for s in man["samples"]}
    keys = [k for k in man["splits"]["train"]][: limit or None]
    model, payload = research_load_checkpoint(ckpt)
    cfg = CaseConfigV1()
    recs, labels, t0 = {}, [], time.perf_counter()
    for i, key in enumerate(keys):
        s = samples[key]
        assert s["split"] == "train", key                         # TRAIN isolation (VAL/TEST never loaded)
        t = quartet_tensor(corpus, s)
        sp = infer_phase(model, t[None])
        ph, _, info = teacher_phase(t[None], steps=steps, init=sp.double())
        q = torch.from_numpy(decode_phase(encode_phase(ph[0, 0].double().numpy())))[None, None].float()
        with torch.no_grad():
            lab = display(multiplane_amplitudes(q, cfg, propagate_fresnel))[0].float()   # [4,1,H,W]
        recs[tensor_key(t)] = {"index": len(labels), "key": key, "target_png_sha256": s["sha256"],
                               "best_step": info["best_step"], "best_objective": info["best_objective"]}
        labels.append(lab)
        if (i + 1) % 32 == 0:
            print(f"{i + 1}/{len(keys)} {time.perf_counter() - t0:.0f}s", flush=True)
    blob = {"manifest_sha256": MANIFEST_SHA, "student_checkpoint_sha256": payload["sha256"], "teacher_steps": steps,
            "records": recs, "labels": torch.stack(labels), "n": len(labels),
            "best_step0_fraction": float(np.mean([r["best_step"] == 0 for r in recs.values()])),
            "elapsed_s": time.perf_counter() - t0}
    torch.save(blob, out)
    return blob


class LabelStore:
    def __init__(self, path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.records, self.labels = blob["records"], blob["labels"]
        self.meta = {k: blob[k] for k in ("manifest_sha256", "student_checkpoint_sha256", "teacher_steps", "n", "best_step0_fraction")}

    def lookup(self, targets: torch.Tensor) -> torch.Tensor:
        """targets [B,4,1,H,W] -> labels [B,4,1,H,W]; raises KeyError on any unlabeled / misaligned quartet."""
        idx = [self.records[tensor_key(targets[b])]["index"] for b in range(targets.shape[0])]
        return self.labels[idx].to(targets.device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus"); ap.add_argument("ckpt"); ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--steps", type=int, default=50)
    a = ap.parse_args()
    b = build(Path(a.corpus), a.ckpt, Path(a.out), a.limit, a.steps)
    print(json.dumps({k: b[k] for k in ("n", "teacher_steps", "best_step0_fraction", "elapsed_s", "student_checkpoint_sha256")}))
