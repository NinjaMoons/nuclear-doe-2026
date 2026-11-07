"""RESEARCH ONLY (R18): matched fine-tune of v2 — arm control (STEM, R3) vs bpmix (STEM + zero-init mixing head, R3).
usage: python -m bpmix_head.train_r18 --arm {control,bpmix} --seed S --updates U --v2 CKPT --manifest M --out NEW_DIR"""
import argparse, functools, hashlib, json, math, time
from pathlib import Path
import torch
import back_projection_stem.train_wrapper as bw
from back_projection_stem.stem import BackProjectionStemUNet
from doe_core.contracts_v1 import CaseConfigV1
from doe_core.fresnel import propagate_fresnel
from doe_models.adam_multiplane import multiplane_amplitudes
from bpmix_head.model import BPMixStemUNet, from_v2
tm = bw.tm; _stock_load = tm.load_checkpoint


def research_load(path):
    hp = torch.load(Path(path), map_location="cpu", weights_only=True).get("hparams", {})
    cls = BPMixStemUNet if hp.get("phase_head") == "bpmix_v0" else (BackProjectionStemUNet if hp.get("stem") == "backprojection_v0" else None)
    if cls is None:
        return _stock_load(path)
    saved = tm.FourInputUNet; tm.FourInputUNet = cls
    try:
        return _stock_load(path)
    finally:
        tm.FourInputUNet = saved


def main():
    ap = argparse.ArgumentParser()
    for a in ("--arm", "--v2", "--manifest", "--out"): ap.add_argument(a, required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--updates", type=int, default=6000); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False); cfg = CaseConfigV1()
    comps = tm.production_components(); data = tm.load_manifest(a.manifest, load_quartet=comps.load_quartet)
    assert data["manifest_sha256"].startswith("76d2474d")
    T = data["splits"]["train"]["targets"]; assert T.shape[0] == 512
    v2, payload = research_load(a.v2); assert payload["sha256"].startswith("19b05dad")
    model = from_v2(v2) if a.arm == "bpmix" else v2
    model.train(); opt = torch.optim.Adam(model.parameters(), lr=1e-3); gen = torch.Generator().manual_seed(30000 + a.seed)
    log, t0 = [], time.perf_counter()
    for step in range(1, a.updates + 1):
        idx = torch.randperm(512, generator=gen)[:4]; batch = T[idx]
        loss, info = bw.research_objective(multiplane_amplitudes(model(batch), cfg, propagate_fresnel), batch, bw.R3_NAME)
        v = float(loss.detach())
        if not math.isfinite(v): raise RuntimeError(f"non-finite at {step}")
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 500 == 0 or step == 1:
            log.append({"step": step, "objective": v, "elapsed_s": time.perf_counter() - t0}); print(log[-1], flush=True)
    ck = out / f"step_{a.updates:06d}.pt"
    tm.save_checkpoint(ck, model, config=cfg, train_log={"partial": True, "arm": a.arm, "seed": a.seed, "updates": a.updates})
    tm.load_checkpoint = research_load
    try:
        s = tm.evaluate_checkpoints([ck], data, comps, out / "eval_val", torch.device("cpu"), ("val",), cfg)
    finally:
        tm.load_checkpoint = _stock_load
    me = Path(__file__)
    json.dump({"eval": s, "log": log, "arm": a.arm, "seed": a.seed, "updates": a.updates, "v2_sha256": payload["sha256"],
               "script_sha256": hashlib.sha256(me.read_bytes()).hexdigest(),
               "model_sha256": hashlib.sha256((me.parent / "model.py").read_bytes()).hexdigest(),
               "train_s": log[-1]["elapsed_s"] if log else None}, open(out / "r18_summary.json", "w"), indent=1)
    v = s["checkpoints"][0]["val"]; print("VAL", a.arm, round(v["pearson_mean"], 5), round(v["ssim_mean"], 5))


if __name__ == "__main__":
    main()
