"""Physics-informed training of FourInputUNet without phase labels (contract v1.1, Issue #5).

model(targets) -> one phase -> the SAME forward/loss as multi-plane Adam (#3):
`multiplane_amplitudes` with an injected propagator, then `multiplane_loss`
(amplitude |U|, per-plane max display, weights 1/4) by default; an opt-in training
objective is selectable (see `training_objective`). No optics is implemented here.
Data loading is not part of this module: callers pass target tensors and IDs.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import math
import time
from numbers import Real
from pathlib import Path
from typing import Sequence

import torch

from doe_core.contracts_v1 import (CONTRACT_VERSION_V1, LOSS_POLICY_V1, OBSERVATION_V1, CaseConfigV1,
                                   FresnelPropagator)
from doe_models.adam_multiplane import multiplane_amplitudes, multiplane_loss
from doe_models.unet_multiplane import FourInputUNet, build_model

__all__ = ["CHECKPOINT_FORMAT", "MANIFEST_SCHEMA", "SPLITS", "TRAINING_OBJECTIVES", "DriverComponents",
           "check_disjoint_ids", "checkpoint_metadata", "evaluate_checkpoint", "evaluate_checkpoints", "evaluate_model", "infer_phase",
           "load_checkpoint", "load_manifest", "main", "physics_loss", "production_components",
           "save_checkpoint", "train_multiplane", "training_objective"]

CHECKPOINT_FORMAT = "doe_fourinput_unet_v1"
CONTRACT_OBJECTIVE = "contract_mse"
LSQ_MIX_OBJECTIVE = "lsq_gain_nmse_plus_0.1_contract_mse_v1"
TRAINING_OBJECTIVES = (CONTRACT_OBJECTIVE, LSQ_MIX_OBJECTIVE)
_LSQ_CONTRACT_WEIGHT = 0.1
_ZERO_TARGET_EPS = 1e-12


def physics_loss(model: torch.nn.Module, targets: torch.Tensor, config: CaseConfigV1,
                 propagate_fn: FresnelPropagator):
    """(total, per_plane[4]) of the model's phase through the shared #3 forward/loss."""
    return multiplane_loss(multiplane_amplitudes(model(targets), config, propagate_fn), targets)


def training_objective(amplitudes: torch.Tensor, targets: torch.Tensor, objective: str = CONTRACT_OBJECTIVE):
    """TRAINING objective only (evaluation/display/export stay v1.1). Returns (value, info).

    `contract_mse`: the #3 contract loss mean((A/max(A) - T)^2), unchanged default.
    `lsq_gain_nmse_plus_0.1_contract_mse_v1` (opt-in, coordinator decision in #5):
    mean over B x 4 planes of min_s ||s*A - T||^2 / ||T||^2 (gain s = <A,T>/<A,A> lives only
    inside this objective, no offset), plus 0.1 * contract MSE; all reductions in float64.
    A plane with ||T||^2 <= 1e-12 uses its per-plane contract MSE instead (counted in info).
    """
    if objective not in TRAINING_OBJECTIVES:
        raise ValueError(f"training objective must be one of {TRAINING_OBJECTIVES}, got {objective!r}")
    if objective == CONTRACT_OBJECTIVE:
        value, _ = multiplane_loss(amplitudes, targets)
        return value, {"contract_mse": float(value.detach()), "lsq_nmse": None, "zero_target_planes": 0}
    a, t = amplitudes.double(), targets.double()
    multiplane_loss(a, t)  # validates shapes and the max(A) > 1e-12 rule of the contract
    peak = a.amax(dim=(-2, -1), keepdim=True)
    contract_plane = ((a / peak - t) ** 2).mean(dim=(-2, -1))                 # [B,4,1]
    t_energy = (t * t).sum(dim=(-2, -1))
    gain = (a * t).sum(dim=(-2, -1)) / (a * a).sum(dim=(-2, -1))
    zero = t_energy <= _ZERO_TARGET_EPS
    safe_energy = torch.where(zero, torch.ones_like(t_energy), t_energy)
    lsq_plane = ((gain[..., None, None] * a - t) ** 2).sum(dim=(-2, -1)) / safe_energy
    lsq_plane = torch.where(zero, contract_plane, lsq_plane)
    value = lsq_plane.mean() + _LSQ_CONTRACT_WEIGHT * contract_plane.mean()
    return value, {"contract_mse": float(contract_plane.detach().mean()), "lsq_nmse": float(lsq_plane.detach().mean()),
                   "zero_target_planes": int(zero.sum())}


_objective = training_objective  # alias: the keyword argument below shadows the name


def evaluate_model(model: torch.nn.Module, targets: torch.Tensor, *, config: CaseConfigV1,
                   propagate_fn: FresnelPropagator, batch_size: int = 8,
                   training_objective: str = CONTRACT_OBJECTIVE, metrics_fn=None) -> dict:
    """Sample-weighted contract loss/per-plane losses and the training objective, no gradients.

    Also returns the collapse reference mean(T^2), the bright fraction of A/max(A) > 0.5 and,
    if `metrics_fn(amplitudes, targets)` is injected (e.g. official #4A export metrics), its dict.
    """
    _check_targets(targets, config)
    objective_name = training_objective
    was_training = model.training
    model.eval()
    total = objective = bright = 0.0
    planes, n, zero, amplitudes = torch.zeros(4, dtype=torch.float64), 0, 0, []
    with torch.no_grad():
        for start in range(0, targets.shape[0], batch_size):
            batch = targets[start:start + batch_size]
            amp = multiplane_amplitudes(model(batch), config, propagate_fn)
            loss, per_plane = multiplane_loss(amp, batch)
            value, info = _objective(amp, batch, objective_name)
            k = batch.shape[0]
            total += float(loss) * k
            objective += float(value) * k
            planes += per_plane.detach().double().cpu() * k
            bright += float((amp / amp.amax(dim=(-2, -1), keepdim=True) > 0.5).double().mean()) * k
            zero += info["zero_target_planes"]
            n += k
            if metrics_fn is not None:
                amplitudes.append(amp)
    model.train(was_training)
    result = {"loss": total / n, "plane_losses": (planes / n).tolist(), "objective": objective / n,
              "training_objective": objective_name, "trivial_mean_t2": float(targets.double().square().mean()),
              "bright_fraction": bright / n, "zero_target_planes": zero}
    if metrics_fn is not None:
        result["metrics"] = metrics_fn(torch.cat(amplitudes), targets)
    return result


def check_disjoint_ids(**splits: Sequence[str] | None) -> None:
    """ValueError if any ID repeats inside a split or appears in two splits."""
    seen: dict[str, str] = {}
    for name, ids in splits.items():
        if ids is None:
            continue
        ids = list(ids)
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate IDs inside split {name!r}")
        for item in ids:
            if item in seen:
                raise ValueError(f"ID {item!r} is in both {seen[item]!r} and {name!r}")
            seen[item] = name


def _check_int(value, name: str, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}")


def _check_targets(targets, config: CaseConfigV1) -> None:
    FourInputUNet._check(targets)  # float dtype, [B,4,1,H,W], finite, range [0,1]
    if tuple(targets.shape[-2:]) != config.image_shape:
        raise ValueError(f"targets must be {config.image_shape} images")


def _check_device(model: torch.nn.Module, *tensors) -> None:
    devices = {p.device for p in model.parameters()} | {t.device for t in tensors if t is not None}
    if len(devices) != 1:
        raise ValueError(f"model parameters and targets must share one device, got {sorted(map(str, devices))}")


def _json_config(config: CaseConfigV1) -> dict:
    return {k: list(v) if isinstance(v, tuple) else v for k, v in dataclasses.asdict(config).items()}


def train_multiplane(model: torch.nn.Module, train_targets: torch.Tensor, *, config: CaseConfigV1,
                     propagate_fn: FresnelPropagator, steps: int, lr: float = 1e-3,
                     batch_size: int = 4, seed: int = 0, val_targets: torch.Tensor | None = None,
                     eval_every: int | None = None, train_ids: Sequence[str] | None = None,
                     val_ids: Sequence[str] | None = None, test_ids: Sequence[str] | None = None,
                     training_objective: str = CONTRACT_OBJECTIVE, val_metrics_fn=None,
                     on_validate=None) -> dict:
    """Train in place with Adam; if `val_targets` is given the best-validation weights are kept.

    `training_objective` selects the TRAINING loss (default: contract MSE); the best state is
    chosen by the same objective on validation, the contract MSE is logged separately.
    `val_metrics_fn(amplitudes, targets)` may inject official export metrics into each val log.
    `on_validate(step, model, entry, is_best, is_last)` is called after every validation with the
    CURRENT weights (before the final best-state restore), e.g. to write periodic/best/last
    checkpoints. One optimizer and one batch generator serve the whole run.
    Model parameters and all target tensors must live on one device (else ValueError).
    Returns a JSON-compatible training log. Batches are drawn with a seed-local generator.
    """
    objective_name = training_objective
    if objective_name not in TRAINING_OBJECTIVES:
        raise ValueError(f"training objective must be one of {TRAINING_OBJECTIVES}, got {objective_name!r}")
    if not isinstance(config, CaseConfigV1):
        raise ValueError("config must be CaseConfigV1")
    _check_int(steps, "steps", 1)
    _check_int(batch_size, "batch_size", 1)
    _check_int(seed, "seed", 0)
    if eval_every is None:
        eval_every = steps
    _check_int(eval_every, "eval_every", 1)
    if isinstance(lr, bool) or not isinstance(lr, Real) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and > 0")
    _check_targets(train_targets, config)
    if val_targets is not None:
        _check_targets(val_targets, config)
    for ids, data, name in ((train_ids, train_targets, "train_ids"), (val_ids, val_targets, "val_ids")):
        if ids is not None and (data is None or len(ids) != data.shape[0]):
            raise ValueError(f"{name} must have one ID per quartet")
    check_disjoint_ids(train=train_ids, val=val_ids, test=test_ids)
    _check_device(model, train_targets, val_targets)
    device = train_targets.device

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    start = time.perf_counter()
    gen = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

    def evaluate(data, metrics_fn=None):
        return evaluate_model(model, data, config=config, propagate_fn=propagate_fn,
                              training_objective=objective_name, metrics_fn=metrics_fn)

    log: dict = {
        "method": "fourinput_unet_physics", "contract_version": CONTRACT_VERSION_V1,
        "observation": OBSERVATION_V1, "loss_policy": LOSS_POLICY_V1, "labels": "none (physics-informed)",
        "training_objective": objective_name,
        "propagator": getattr(propagate_fn, "__qualname__", type(propagate_fn).__qualname__),
        "hparams": getattr(model, "hparams", None), "config": _json_config(config),
        "steps": steps, "lr": float(lr), "batch_size": batch_size, "seed": seed, "eval_every": eval_every,
        "n_train": int(train_targets.shape[0]), "n_val": 0 if val_targets is None else int(val_targets.shape[0]),
        "train_ids": None if train_ids is None else list(train_ids),
        "val_ids": None if val_ids is None else list(val_ids),
        "test_ids": None if test_ids is None else list(test_ids),
        "initial": None, "history": [], "val": [],
        "best_step": None, "best_val_loss": None, "zero_target_fallbacks": 0,
        "device": str(device), "dtype": str(train_targets.dtype).replace("torch.", ""),
        "torch_version": str(torch.__version__),
    }
    initial = evaluate(train_targets)
    log["initial"] = {"step": 0, "train_loss": initial["loss"], "plane_losses": initial["plane_losses"],
                      "objective": initial["objective"], "trivial_mean_t2": initial["trivial_mean_t2"],
                      "bright_fraction": initial["bright_fraction"]}
    initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_state = None

    def validate(step: int) -> None:
        nonlocal best_state
        r = evaluate(val_targets, val_metrics_fn)
        entry = {"step": step, "val_loss": r["loss"], "contract_mse": r["loss"], "objective": r["objective"],
                 "plane_losses": r["plane_losses"], "trivial_mean_t2": r["trivial_mean_t2"],
                 "bright_fraction": r["bright_fraction"], "zero_target_planes": r["zero_target_planes"],
                 "metrics": r.get("metrics")}
        log["val"].append(entry)
        is_best = log["best_val_loss"] is None or r["objective"] < log["best_val_loss"]
        if is_best:
            log["best_step"], log["best_val_loss"] = step, r["objective"]
            best_state = copy.deepcopy(model.state_dict())
        if on_validate is not None:
            on_validate(step, model, entry, is_best, step == steps)

    if val_targets is not None:
        validate(0)
    model.train()
    n = train_targets.shape[0]
    for step in range(1, steps + 1):
        sync()
        t_step = time.perf_counter()
        idx = torch.randperm(n, generator=gen)[:min(batch_size, n)]
        batch = train_targets[idx.to(device)]
        optimizer.zero_grad(set_to_none=True)
        loss, info = _objective(
            multiplane_amplitudes(model(batch), config, propagate_fn), batch, objective_name)
        value = float(loss.detach())
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite training objective at step {step}")
        loss.backward()
        grad_norm = math.sqrt(sum(float(p.grad.detach().double().square().sum())
                                  for p in model.parameters() if p.grad is not None))
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient at step {step}")
        optimizer.step()
        sync()
        log["zero_target_fallbacks"] += info["zero_target_planes"]
        log["history"].append({"step": step, "train_loss": value, "objective": value,
                               "contract_mse": info["contract_mse"], "grad_norm": grad_norm,
                               "step_s": time.perf_counter() - t_step})
        if val_targets is not None and (step % eval_every == 0 or step == steps):
            validate(step)

    if best_state is not None:
        model.load_state_dict(best_state)
    final_state = model.state_dict()
    log["weights_delta_l2"] = math.sqrt(sum(
        float((final_state[k].double() - v.double()).square().sum())
        for k, v in initial_state.items() if v.is_floating_point()))
    log["elapsed_s"] = time.perf_counter() - start
    log["timing_scope"] = "initial eval + training + validation evals"
    return log


def infer_phase(model: torch.nn.Module, targets: torch.Tensor) -> torch.Tensor:
    """Detached phase [B,1,H,W] from the model in eval mode; the caller's mode is restored."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return model(targets).detach()
    finally:
        model.train(was_training)


def save_checkpoint(path: str | Path, model: FourInputUNet, *, config: CaseConfigV1,
                    train_log: dict, code_commit: str | None = None, extra: dict | None = None) -> str:
    """Save weights + hparams + config + log + code commit (plain types only); return SHA256."""
    path = Path(path)
    payload = {
        "format": CHECKPOINT_FORMAT, "contract_version": CONTRACT_VERSION_V1,
        "hparams": model.hparams, "dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
        "state_dict": model.state_dict(), "config": _json_config(config),
        "train_log": train_log, "code_commit": code_commit,
        "torch_version": str(torch.__version__), "extra": extra or {},
    }
    torch.save(payload, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


_CHECKPOINT_DTYPES = {"float32": torch.float32, "float64": torch.float64}


def load_checkpoint(path: str | Path) -> tuple[FourInputUNet, dict]:
    """Strictly load a checkpoint of THIS contract; payload gains its file `sha256`.

    `torch.load(weights_only=True)` (no arbitrary pickle). Before restoring weights the
    format, contract_version, full case config, every hparam (in_planes/head/norm too)
    and dtype must match the current code exactly; otherwise ValueError. The model is
    constructed under `fork_rng`, so loading does not change the global RNG.
    """
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    if payload.get("contract_version") != CONTRACT_VERSION_V1:
        raise ValueError(f"checkpoint contract {payload.get('contract_version')!r} != {CONTRACT_VERSION_V1!r}")
    if payload.get("config") != _json_config(CaseConfigV1()):
        raise ValueError("checkpoint config does not match the official CaseConfigV1")
    dtype = _CHECKPOINT_DTYPES.get(payload.get("dtype"))
    if dtype is None:
        raise ValueError(f"checkpoint dtype {payload.get('dtype')!r} is not float32/float64")
    hp = payload.get("hparams")
    if not isinstance(hp, dict):
        raise ValueError("checkpoint hparams missing")
    with torch.random.fork_rng(devices=[]):
        model = FourInputUNet(base_channels=hp.get("base_channels"), depth=hp.get("depth"))
    if model.hparams != hp:
        raise ValueError(f"checkpoint hparams {hp} do not match this architecture {model.hparams}")
    model = model.to(dtype=dtype)
    state = payload.get("state_dict")
    # Every entry of this architecture is a finite floating tensor of the checkpoint dtype;
    # anything else (int tensors, non-tensors, NaN/Inf) is rejected, never cast or loaded.
    if not isinstance(state, dict) or any(not isinstance(v, torch.Tensor) or v.dtype != dtype
                                          for v in state.values()):
        raise ValueError("checkpoint state_dict missing, non-tensor or of a different dtype")
    bad = [k for k, v in state.items() if not bool(torch.isfinite(v).all())]
    if bad:
        raise ValueError(f"checkpoint weights contain NaN/Inf: {bad}")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"checkpoint weights do not fit the architecture: {exc}") from None
    payload["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return model, payload


def checkpoint_metadata(payload: dict) -> dict:
    """JSON-compatible provenance of a loaded checkpoint, e.g. for export_run(metadata=...)."""
    log = payload.get("train_log") or {}
    keys = ("method", "propagator", "steps", "lr", "batch_size", "seed", "n_train", "n_val",
            "train_ids", "val_ids", "test_ids", "best_step", "best_val_loss", "weights_delta_l2",
            "observation", "loss_policy")
    return {
        "method": "fourinput_unet", "checkpoint_format": payload["format"],
        "checkpoint_sha256": payload.get("sha256"), "code_commit": payload.get("code_commit"),
        "contract_version": payload["contract_version"], "hparams": payload["hparams"],
        "dtype": payload["dtype"], "torch_version": payload.get("torch_version"),
        "training": {k: log.get(k) for k in keys},
    }


# ============================================================================ training driver (#5B)
# python -m doe_models.train_multiplane --manifest M --out-dir NEW_DIR --device cuda [...]
# One continuous run (one Adam, one batch generator), validation at 0 / every N / final, weights-only
# checkpoints (periodic, best by validation objective, last), then a NEW-PROCESS evaluation of the best
# checkpoint through the existing export_run (#7). Physics, PNG IO, metrics and export are injected.
MANIFEST_SCHEMA = "doe_training_manifest_v1"
SPLITS = ("train", "val", "test")
_CODE_FILES = ("doe_models/train_multiplane.py", "doe_models/unet_multiplane.py", "doe_models/adam_multiplane.py",
               "doe_core/contracts_v1.py", "doe_core/observables.py", "doe_core/fresnel.py")


@dataclasses.dataclass(frozen=True)
class DriverComponents:
    """Injected collaborators: Fresnel (#14), PNG quartet loader (#4A), val metrics (#4A), export_run (#7)."""
    propagate: FresnelPropagator
    load_quartet: object
    val_metrics: object
    export_run: object


def production_components() -> DriverComponents:
    """The project's real modules; imported lazily so the trainer itself has no hard dependency."""
    import numpy as np
    from doe_core.fresnel import propagate_fresnel
    from doe_experiments.data import load_png_quartet
    from doe_experiments.export import export_run
    from doe_experiments.metrics import evaluate_multiplane
    from doe_experiments.png_io import amplitude_to_u8

    def val_metrics(amplitudes, targets):
        codes = amplitude_to_u8(amplitudes.detach().cpu())
        pairs = evaluate_multiplane(torch.from_numpy(codes.astype(np.float64) / 255), targets.detach().cpu().double())
        P = np.array([[np.nan if m.pearson is None else m.pearson for m in b] for b in pairs], dtype=float)
        S = np.array([[m.ssim for m in b] for b in pairs], dtype=float)
        return {"pearson_plane": [_finite_or_none(v) for v in np.nanmean(P, 0)],
                "ssim_plane": [float(v) for v in S.mean(0)], "source": "official amplitude_to_u8 + evaluate_multiplane"}

    return DriverComponents(propagate_fresnel, load_png_quartet, val_metrics, export_run)


def _finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _safe_relative(base: Path, rel) -> Path:
    if not isinstance(rel, str) or not rel:
        raise ValueError("manifest paths must be non-empty strings")
    p = Path(rel)
    if p.is_absolute() or p.drive or ".." in p.parts:
        raise ValueError(f"manifest path {rel!r} must be relative to the manifest and stay inside it")
    full = (base / p).resolve()
    if base.resolve() not in full.parents:
        raise ValueError(f"manifest path {rel!r} escapes the manifest directory")
    return full


def load_manifest(path, *, load_quartet) -> dict:
    """Strictly validate a `doe_training_manifest_v1` and load each split with `load_quartet`.

    Every PNG must match its SHA256; keys `corpus_id:local_sample_id` are unique; one origin_id never
    appears in two splits; train/val/test are all non-empty; paths stay inside the manifest directory.
    """
    import json
    path = Path(path)
    raw = path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA!r}")
    if manifest.get("contract_version") != CONTRACT_VERSION_V1:
        raise ValueError(f"manifest contract_version must be {CONTRACT_VERSION_V1!r}")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("manifest samples must be a non-empty list")
    base = path.parent
    keys, origin_split = set(), {}
    rows = {s: [] for s in SPLITS}
    for i, s in enumerate(samples):
        if not isinstance(s, dict):
            raise ValueError(f"sample {i} must be an object")
        corpus, local, origin, split = s.get("corpus_id"), s.get("local_sample_id"), s.get("origin_id"), s.get("split")
        for name, value in (("corpus_id", corpus), ("local_sample_id", local), ("origin_id", origin)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"sample {i}: {name} must be a non-empty string")
        if ":" in corpus or ":" in local:
            raise ValueError(f"sample {i}: corpus_id/local_sample_id must not contain ':'")
        if split not in SPLITS:
            raise ValueError(f"sample {i}: split must be one of {SPLITS}, got {split!r}")
        key = f"{corpus}:{local}"
        if key in keys:
            raise ValueError(f"duplicate sample key {key!r}")
        keys.add(key)
        if origin_split.setdefault(origin, split) != split:
            raise ValueError(f"origin {origin!r} appears in splits {origin_split[origin]!r} and {split!r}")
        paths, hashes = s.get("paths"), s.get("sha256")
        if not isinstance(paths, list) or len(paths) != 4 or not isinstance(hashes, list) or len(hashes) != 4:
            raise ValueError(f"sample {key}: exactly 4 ordered paths and 4 sha256 values are required")
        files = [_safe_relative(base, p) for p in paths]
        for f, h in zip(files, hashes):
            if not f.is_file():
                raise ValueError(f"sample {key}: missing file {f}")
            if not isinstance(h, str) or _sha256_file(f) != h.lower():
                raise ValueError(f"sample {key}: SHA256 mismatch for {f.name}")
        rows[split].append((key, origin, s.get("kind"), files))
    for split in SPLITS:
        if not rows[split]:
            raise ValueError(f"manifest split {split!r} is empty")
    splits = {}
    for split, items in rows.items():
        targets = torch.cat([load_quartet(files) for _, _, _, files in items])
        splits[split] = {"targets": targets, "ids": [k for k, _, _, _ in items],
                         "origins": [o for _, o, _, _ in items], "kinds": [kd for _, _, kd, _ in items],
                         "paths": [files for _, _, _, files in items]}
    return {"splits": splits, "manifest_sha256": hashlib.sha256(raw).hexdigest(), "manifest_path": str(path.resolve()),
            "n_samples": len(samples), "synthetic": bool(manifest.get("synthetic", True)),
            "corpus_ids": sorted({k.split(":", 1)[0] for k in keys})}


def _code_provenance() -> dict:
    import subprocess
    root = Path(__file__).resolve().parents[1]
    shas = {rel: _sha256_file(root / rel) for rel in _CODE_FILES if (root / rel).is_file()}
    commit = dirty = None
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=20, check=True).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, timeout=20, check=True).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return {"code_commit": commit, "git_dirty": dirty, "code_sha256": shas}


def _environment(device: torch.device) -> dict:
    import platform
    import sys
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": str(torch.__version__),
           "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
           "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
           "cpu_threads": torch.get_num_threads(), "device": str(device)}
    for mod in ("numpy", "PIL"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - environment report only
            env[mod] = None
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        env.update({"gpu_name": props.name, "gpu_total_memory_bytes": int(props.total_memory),
                    "gpu_capability": f"{props.major}.{props.minor}"})
    return env


def _write_json(path: Path, value) -> None:
    import json
    path.write_text(json.dumps(value, indent=1, allow_nan=False), encoding="utf-8")


def evaluate_checkpoint(checkpoint, data: dict, components: DriverComponents, out_dir: Path,
                        device: torch.device, config: CaseConfigV1 | None = None, splits=("val",)) -> dict:
    """Load `checkpoint` and run export_run for every quartet of `splits`; aggregate the 8 metrics.

    Default is VAL only. Test must be requested explicitly, once, for a checkpoint chosen on VAL.
    """
    import json
    import numpy as np
    from PIL import Image
    config = config or CaseConfigV1()
    model, payload = load_checkpoint(checkpoint)
    model = model.to(device)
    metadata = checkpoint_metadata(payload)
    extra = payload.get("extra") or {}
    name = Path(checkpoint).name
    summary = {"name": name, "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": payload["sha256"],
               "kind": extra.get("kind"), "step": extra.get("step"), "manifest_sha256": data["manifest_sha256"]}
    for split in splits:
        if split not in ("val", "test"):
            raise ValueError(f"evaluation split must be val or test, got {split!r}")
        part = data["splits"][split]
        phase = infer_phase(model, part["targets"].to(device)).cpu()
        P, S, lit, tlit, peak = [], [], [], [], []
        for i, (key, files) in enumerate(zip(part["ids"], part["paths"])):
            run = out_dir / "export" / name / split / f"{i:03d}"
            run.parent.mkdir(parents=True, exist_ok=True)
            components.export_run(files, phase[i:i + 1], run, synthetic_test=data["synthetic"],
                                  metadata={**metadata, "sample_key": key, "origin_id": part["origins"][i],
                                            "split": split, "manifest_sha256": data["manifest_sha256"]})
            planes = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["planes"]
            P.append([p["pearson"] if p.get("pearson_defined") else np.nan for p in planes])
            S.append([p["ssim"] for p in planes])
            rec = np.stack([np.array(Image.open(run / f"reconstructed_{k:02d}.png")) for k in range(1, 5)]) / 255.0
            tgt = part["targets"][i, :, 0].double().numpy()
            lit.append((rec > 0.5).mean(axis=(1, 2)))
            tlit.append((tgt > 0.5).mean(axis=(1, 2)))
            amp = np.load(run / "raw.npz")["amplitude_quantized"].reshape(4, -1).astype(np.float64)
            peak.append(amp.max(1) ** 2 / (amp ** 2).sum(1))
        P, S = np.array(P, dtype=float), np.array(S, dtype=float)
        pp, sp = np.nanmean(P, 0), S.mean(0)
        lit_frac, target_lit = float(np.mean(lit)), float(np.mean(tlit))
        summary[split] = {
            "n": len(part["ids"]), "pearson_plane": [_finite_or_none(v) for v in pp], "ssim_plane": sp.tolist(),
            "pearson_mean": _finite_or_none(np.nanmean(P)), "ssim_mean": float(S.mean()),
            "worst_plane_pearson": _finite_or_none(np.nanmin(pp)), "worst_plane_ssim": float(sp.min()),
            "worst_pair_pearson": _finite_or_none(np.nanmin(P)), "undefined_pearson": int(np.isnan(P).sum()),
            "lit_frac": lit_frac, "target_lit_frac": target_lit, "peak_pixel_power_frac": float(np.mean(peak)),
            "collapse": bool(lit_frac < 0.2 * target_lit or not (np.nanmean(P) >= 0.05)),
            "collapse_rule": "lit fraction (PNG>0.5) < 0.2 * target, or mean Pearson < 0.05 (diagnostic, not a hard gate)"}
    return summary


def evaluate_checkpoints(checkpoints, data: dict, components: DriverComponents, out_dir: Path,
                         device: torch.device, splits=("val",), config: CaseConfigV1 | None = None) -> dict:
    """Evaluate several checkpoints; report best-by-objective and best-by-VAL-Pearson SEPARATELY."""
    rows = [evaluate_checkpoint(c, data, components, out_dir, device, config, splits) for c in checkpoints]
    by_objective = next(({"name": r["name"], "step": r["step"]} for r in rows if r["kind"] == "best"), None)
    by_pearson = None
    if "val" in splits:
        scored = [r for r in rows if r["val"]["pearson_mean"] is not None]
        if scored:
            top = max(scored, key=lambda r: r["val"]["pearson_mean"])
            by_pearson = {"name": top["name"], "step": top["step"], "pearson_mean": top["val"]["pearson_mean"]}
    return {"splits": list(splits), "checkpoints": rows, "best_by_objective": by_objective,
            "best_by_val_pearson": by_pearson, "manifest_sha256": data["manifest_sha256"],
            "evaluation": "export_run per quartet: phase -> DOE.png -> reload -> 4 Fresnel -> 4 PNG -> reload -> Pearson/SSIM",
            "note": ("best_by_objective is the training selection; best_by_val_pearson is a separate VAL report. "
                     "Test is opened only explicitly (--eval-splits test) for a checkpoint fixed on VAL.")}


def _parser():
    import argparse
    ap = argparse.ArgumentParser(prog="python -m doe_models.train_multiplane",
                                 description="Train FourInputUNet on a doe_training_manifest_v1 (one continuous run).")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path, help="new directory; an existing one is refused")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=100, help="validation AND periodic checkpoint interval")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-channels", type=int, default=16)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--objective", default=LSQ_MIX_OBJECTIVE, choices=TRAINING_OBJECTIVES)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--smoke-steps", type=int, default=0, help="only time K real train steps; no checkpoints")
    ap.add_argument("--evaluate-checkpoint", type=Path, nargs="+", default=None,
                    help="only evaluate these checkpoints through export_run (no training)")
    ap.add_argument("--eval-splits", nargs="+", choices=("val", "test"), default=["val"],
                    help="splits for --evaluate-checkpoint; default VAL only (open test once, after selection)")
    ap.add_argument("--no-eval", action="store_true", help="skip the post-run new-process VAL evaluation")
    return ap


def main(argv=None, components: DriverComponents | None = None) -> int:
    """Entrypoint. `components` is for tests; by default the project's real modules are used."""
    import subprocess
    import sys
    args = _parser().parse_args(argv)
    injected = components is not None
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested (--device cuda) but torch.cuda.is_available() is False")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    components = components or production_components()
    config = CaseConfigV1()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)                       # never overwrite an earlier run
    data = load_manifest(args.manifest, load_quartet=components.load_quartet)
    env = _environment(device)
    _write_json(out / "environment.json", env)

    if args.evaluate_checkpoint is not None:
        _write_json(out / "eval_summary.json", evaluate_checkpoints(
            args.evaluate_checkpoint, data, components, out, device, tuple(args.eval_splits), config))
        return 0

    sp = data["splits"]
    run_config = {
        "driver": "python -m doe_models.train_multiplane", "argv": list(argv) if argv is not None else sys.argv[1:],
        "training_objective": args.objective, "updates": args.updates, "eval_every": args.eval_every,
        "checkpoint_every": args.eval_every, "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
        "hparams": {"base_channels": args.base_channels, "depth": args.depth}, "optimizer": "Adam (defaults)",
        "precision": "float32", "amp": False, "selection": "best validation objective (same as training objective)",
        "checkpoints": "weights only, NOT resumable (no optimizer/RNG state)", "smoke_steps": args.smoke_steps,
        "manifest_sha256": data["manifest_sha256"], "manifest_path": data["manifest_path"],
        "split_sizes": {k: len(v["ids"]) for k, v in sp.items()}, "corpus_ids": data["corpus_ids"],
        "synthetic": data["synthetic"], "contract_version": CONTRACT_VERSION_V1, **_code_provenance()}
    _write_json(out / "run_config.json", run_config)

    train, val = sp["train"]["targets"].to(device), sp["val"]["targets"].to(device)
    model = build_model(seed=args.seed, base_channels=args.base_channels, depth=args.depth).to(device)
    _check_device(model, train, val)
    common = dict(config=config, propagate_fn=components.propagate, lr=args.lr, batch_size=args.batch_size,
                  seed=args.seed, training_objective=args.objective)

    if args.smoke_steps:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        log = train_multiplane(model, train, steps=args.smoke_steps, **common)
        times = [h["step_s"] for h in log["history"]]
        steady = times[1:] or times
        _write_json(out / "smoke.json", {
            "steps": args.smoke_steps, "device": str(device), "s_per_step": sum(steady) / len(steady),
            "first_step_s": times[0], "step_s": times, "objective_first_last": [log["history"][0]["objective"],
                                                                                  log["history"][-1]["objective"]],
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "timing": "per optimizer update incl. forward/backward/step; cuda synchronized", **env})
        return 0

    ckdir = out / "checkpoints"
    ckdir.mkdir()
    run_config_sha = _sha256_file(out / "run_config.json")

    def extra(kind: str, step: int) -> dict:
        return {"kind": kind, "step": step, "resumable": False, "note": "weights only; not a resumable training state",
                "manifest_sha256": data["manifest_sha256"], "run_config_sha256": run_config_sha}

    def on_validate(step, m, entry, is_best, is_last):
        partial = {"partial": True, "step": step, "val_entry": entry, "training_objective": args.objective}
        if step > 0:
            save_checkpoint(ckdir / f"step_{step:06d}.pt", m, config=config, train_log=partial,
                            code_commit=run_config["code_commit"], extra=extra("periodic", step))
        if is_best:
            save_checkpoint(ckdir / "best.pt", m, config=config, train_log=partial,
                            code_commit=run_config["code_commit"], extra=extra("best", step))
        if is_last:
            save_checkpoint(ckdir / "last.pt", m, config=config, train_log=partial,
                            code_commit=run_config["code_commit"], extra=extra("last", step))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    log = train_multiplane(model, train, steps=args.updates, val_targets=val, eval_every=args.eval_every,
                           train_ids=sp["train"]["ids"], val_ids=sp["val"]["ids"], test_ids=sp["test"]["ids"],
                           val_metrics_fn=components.val_metrics, on_validate=on_validate, **common)
    # the model now holds the best-validation weights: rewrite best.pt with the complete log
    best_sha = save_checkpoint(ckdir / "best.pt", model, config=config, train_log=log,
                               code_commit=run_config["code_commit"], extra=extra("best", log["best_step"]))
    log["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    log["checkpoints"] = {p.name: _sha256_file(p) for p in sorted(ckdir.glob("*.pt"))}
    log["best_checkpoint_sha256"] = best_sha
    _write_json(out / "train_log.json", log)

    if not args.no_eval:   # VAL only, every checkpoint (periodic, best, last); test is NOT opened here
        checkpoints = sorted(ckdir.glob("*.pt"))
        if injected:
            summary = evaluate_checkpoints(checkpoints, data, components, out / "eval_val", device, ("val",), config)
        else:   # production: a NEW Python process loads the checkpoints (weights_only) and exports
            subprocess.run([sys.executable, "-m", "doe_models.train_multiplane", "--manifest", str(args.manifest),
                            "--out-dir", str(out / "eval_val"), "--device", args.device, "--eval-splits", "val",
                            "--evaluate-checkpoint", *map(str, checkpoints)], check=True)
            import json
            summary = json.loads((out / "eval_val" / "eval_summary.json").read_text(encoding="utf-8"))
        _write_json(out / "eval_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
