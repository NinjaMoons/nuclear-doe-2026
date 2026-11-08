"""RESEARCH ONLY (R25, NIH-POST-R25-GF-DDFN-037): matched fine-tune of STEM V2, arm control (STEM, R3) vs dpsi
(STEM + zero-init GF-DDFN target-plane phase dPsi_k(T), R3). Everything else identical: V2 init 19b05dad, TRAIN manifest
76d2474d (no TEST decoded; see run_r25_notest.py), sampler seed, Adam lr 1e-3, batch 4, R3 objective, update budget,
official evaluate_checkpoints (saved+reloaded L8 DOE.png and four L8 amplitude PNGs) at ONE pre-fixed endpoint.
--subset64: train only on 64 TRAIN quartets, 32 A + 32 B, drawn once with torch seed `--seed` (same for both arms).
usage: python -m r25_gf_ddfn.run_r25_notest --arm {control,dpsi} --seed 2026 --updates U --v2 CKPT --manifest M --out NEW_DIR
       [--device cuda] [--subset64] [--no-eval]"""
import argparse, hashlib, json, math, time
from pathlib import Path

import torch

import back_projection_stem.train_wrapper as bw
from back_projection_stem.stem import BackProjectionStemUNet
from doe_core.contracts_v1 import CaseConfigV1
from doe_core.fresnel import propagate_fresnel
from doe_models.adam_multiplane import multiplane_amplitudes
from r25_gf_ddfn.model_r25 import DPsiStemUNet, from_v2

tm = bw.tm
_stock_load = tm.load_checkpoint


def research_load(path):
    hp = torch.load(Path(path), map_location="cpu", weights_only=True).get("hparams", {})
    cls = DPsiStemUNet if hp.get("phase_head") == "r25_dpsi_v0" else (BackProjectionStemUNet if hp.get("stem") == "backprojection_v0" else None)
    if cls is None:
        return _stock_load(path)
    saved = tm.FourInputUNet; tm.FourInputUNet = cls
    try:
        return _stock_load(path)
    finally:
        tm.FourInputUNet = saved


def subset64(train, seed):
    grp = list(train["kinds"])  # manifest field `kind` in {"A", "B"} (256 + 256 in TRAIN512)
    assert set(grp) == {"A", "B"}, set(grp)
    g = torch.Generator().manual_seed(seed)
    pick = []
    for G in ("A", "B"):
        pool = [j for j, x in enumerate(grp) if x == G]
        assert len(pool) >= 32, (G, len(pool))
        pick += [pool[j] for j in torch.randperm(len(pool), generator=g)[:32].tolist()]
    return sorted(pick), grp


def main():
    ap = argparse.ArgumentParser()
    for a in ("--arm", "--v2", "--manifest", "--out"):
        ap.add_argument(a, required=True)
    ap.add_argument("--seed", type=int, default=2026); ap.add_argument("--updates", type=int, default=100)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-eval", action="store_true"); ap.add_argument("--subset64", action="store_true")
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    assert a.arm in ("control", "dpsi")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False); cfg = CaseConfigV1()
    comps = tm.production_components(); data = tm.load_manifest(a.manifest, load_quartet=comps.load_quartet)
    assert data["manifest_sha256"].startswith("76d2474d")
    tr = data["splits"]["train"]; T = tr["targets"]; assert T.shape[0] == 512
    pool = list(range(512)); grp = None
    if a.subset64:
        pool, grp = subset64(tr, a.seed)
    v2, payload = research_load(a.v2); assert payload["sha256"].startswith("19b05dad")
    model = from_v2(v2) if a.arm == "dpsi" else v2
    dev = torch.device(a.device); model = model.to(dev); model._config = cfg
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    torch.manual_seed(a.seed)
    model.train(); opt = torch.optim.Adam(model.parameters(), lr=1e-3); gen = torch.Generator().manual_seed(30000 + a.seed)
    P = torch.tensor(pool)
    log, t0 = [], time.perf_counter()
    for step in range(1, a.updates + 1):
        idx = P[torch.randperm(len(pool), generator=gen)[:4]]; batch = T[idx].to(dev)
        loss, info = bw.research_objective(multiplane_amplitudes(model(batch), cfg, propagate_fresnel), batch, bw.R3_NAME)
        v = float(loss.detach())
        if not math.isfinite(v):
            raise RuntimeError(f"non-finite at {step}")
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 500 == 0 or step in (1, 10, 50, 100):
            rec = {"step": step, "objective": v, "elapsed_s": time.perf_counter() - t0}
            if a.arm == "dpsi":
                with torch.no_grad():
                    rec["dpsi_abs_mean"] = float(model.target_phase(batch).abs().mean())
            log.append(rec); print(rec, flush=True)
    ck = out / f"step_{a.updates:06d}.pt"
    tm.save_checkpoint(ck, model, config=cfg, train_log={"partial": True, "arm": a.arm, "seed": a.seed, "updates": a.updates,
                                                         "subset64": a.subset64})
    ck_sha = hashlib.sha256(ck.read_bytes()).hexdigest()
    # reload check: research_load must rebuild the same class and give bit-identical outputs
    m2, _ = research_load(ck); m2 = m2.to(dev).eval(); model.eval()
    with torch.no_grad():
        xb = T[:4].to(dev); reload_maxdiff = float((m2(xb) - model(xb)).abs().max())
    me = Path(__file__)
    summ = {"arm": a.arm, "seed": a.seed, "updates": a.updates, "subset64": a.subset64, "train_pool": pool if a.subset64 else "all512",
            "device": a.device, "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
            "peak_vram_mb": torch.cuda.max_memory_allocated(dev) / 2**20 if dev.type == "cuda" else None,
            "v2_sha256": payload["sha256"], "ckpt": str(ck), "ckpt_sha256": ck_sha, "reload_max_abs_diff": reload_maxdiff,
            "script_sha256": hashlib.sha256(me.read_bytes()).hexdigest(),
            "model_sha256": hashlib.sha256((me.parent / "model_r25.py").read_bytes()).hexdigest(),
            "train_s": time.perf_counter() - t0, "log": log}
    if not a.no_eval:
        tm.load_checkpoint = research_load
        try:
            summ["eval"] = tm.evaluate_checkpoints([ck], data, comps, out / "eval_val", dev, ("val",), cfg)
        finally:
            tm.load_checkpoint = _stock_load
        v = summ["eval"]["checkpoints"][0]["val"]; print("VAL", a.arm, round(v["pearson_mean"], 5), round(v["ssim_mean"], 5))
    json.dump(summ, open(out / "r25_summary.json", "w"), indent=1)
    print("DONE", a.arm, "ckpt", ck_sha[:12], "reload_diff", reload_maxdiff, "peak_vram_mb", summ["peak_vram_mb"], "train_s", round(summ["train_s"], 1))


if __name__ == "__main__":
    main()
