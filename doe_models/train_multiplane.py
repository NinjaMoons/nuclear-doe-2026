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
from doe_models.unet_multiplane import FourInputUNet

__all__ = ["CHECKPOINT_FORMAT", "TRAINING_OBJECTIVES", "check_disjoint_ids", "checkpoint_metadata",
           "evaluate_model", "infer_phase", "load_checkpoint", "physics_loss", "save_checkpoint",
           "train_multiplane", "training_objective"]

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


def _json_config(config: CaseConfigV1) -> dict:
    return {k: list(v) if isinstance(v, tuple) else v for k, v in dataclasses.asdict(config).items()}


def train_multiplane(model: torch.nn.Module, train_targets: torch.Tensor, *, config: CaseConfigV1,
                     propagate_fn: FresnelPropagator, steps: int, lr: float = 1e-3,
                     batch_size: int = 4, seed: int = 0, val_targets: torch.Tensor | None = None,
                     eval_every: int | None = None, train_ids: Sequence[str] | None = None,
                     val_ids: Sequence[str] | None = None, test_ids: Sequence[str] | None = None,
                     training_objective: str = CONTRACT_OBJECTIVE, val_metrics_fn=None) -> dict:
    """Train in place with Adam; if `val_targets` is given the best-validation weights are kept.

    `training_objective` selects the TRAINING loss (default: contract MSE); the best state is
    chosen by the same objective on validation, the contract MSE is logged separately.
    `val_metrics_fn(amplitudes, targets)` may inject official export metrics into each val log.
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
        "device": str(train_targets.device), "dtype": str(train_targets.dtype).replace("torch.", ""),
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
        if log["best_val_loss"] is None or r["objective"] < log["best_val_loss"]:
            log["best_step"], log["best_val_loss"] = step, r["objective"]
            best_state = copy.deepcopy(model.state_dict())

    if val_targets is not None:
        validate(0)
    model.train()
    n = train_targets.shape[0]
    for step in range(1, steps + 1):
        idx = torch.randperm(n, generator=gen)[:min(batch_size, n)]
        batch = train_targets[idx.to(train_targets.device)]
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
        log["zero_target_fallbacks"] += info["zero_target_planes"]
        log["history"].append({"step": step, "train_loss": value, "objective": value,
                               "contract_mse": info["contract_mse"], "grad_norm": grad_norm})
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
