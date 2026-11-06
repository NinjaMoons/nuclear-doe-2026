"""One saved DOE is the source of all four submitted reconstructions."""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import time
from typing import Callable

import numpy as np
import torch

from doe_core.contracts_v1 import (CaseConfigV1, FresnelField,
    RECONSTRUCTION_POLICY_V1, SSIM_POLICY_V1, CONTRACT_VERSION_V1, OBSERVATION_V1)
from doe_core.observables import FieldObservables, field_observables
from doe_experiments.phase_codec import load_doe_png, save_doe_png


@dataclass(frozen=True)
class ExportComponents:
    propagate: Callable
    load_quartet: Callable
    amplitude_to_u8: Callable
    save_reconstructions: Callable
    evaluate: Callable


def production_components() -> ExportComponents:
    # No fallback simulator, stub, codec or metric is selected in production.
    from doe_core.fresnel import propagate_fresnel
    from doe_experiments.data import load_png_quartet
    from doe_experiments.png_io import amplitude_to_u8, save_reconstructions
    from doe_experiments.metrics import evaluate_multiplane
    return ExportComponents(propagate_fresnel, load_png_quartet,
                            amplitude_to_u8, save_reconstructions, evaluate_multiplane)


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                    encoding="utf-8")


def _git_sha() -> str | None:
    result = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]),
                             "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _quartet(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.shape != (1, 4, 1, 128, 128):
        raise ValueError("one quartet must have shape [1,4,1,128,128]")
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError("quartet must be float32/64")
    if not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any():
        raise ValueError("quartet values must be finite in [0,1]")
    return value


def _reconstruct(phase: torch.Tensor, config: CaseConfigV1,
                 propagate: Callable) -> FieldObservables:
    field = torch.exp(1j * phase)  # Explicit illumination A=1, same field for all z.
    amplitudes, intensities = [], []
    for index in range(4):
        geometry = config.plane(index)
        result = propagate(field, geometry)
        if not isinstance(result, FresnelField) or result.field.shape != field.shape:
            raise ValueError("propagator must return matching FresnelField")
        if result.field.dtype != field.dtype or result.field.device != field.device:
            raise ValueError("propagator changed field dtype/device")
        expected_pitch = np.array(geometry.output_pitch_m(128, 128))
        if np.shape(result.output_pitch_m) != (2,) or not np.allclose(result.output_pitch_m, expected_pitch, rtol=1e-12, atol=0):
            raise ValueError("wrong native output sampling")
        if result.distance_m != geometry.distance_m:
            raise ValueError("propagator returned the wrong plane distance")
        measurement = field_observables(result.field)
        amplitudes.append(measurement.amplitude)
        intensities.append(measurement.intensity)
    return FieldObservables(torch.stack(amplitudes, dim=1), torch.stack(intensities, dim=1))


def _display(raw: torch.Tensor, conversion: Callable) -> tuple[np.ndarray, list[float], torch.Tensor]:
    values = raw.detach().cpu().numpy().astype(np.float64)
    maxima = values.max(axis=(-2, -1), keepdims=True)
    if not np.isfinite(values).all() or (values < 0).any() or (maxima <= 1e-12).any():
        raise ValueError("invalid or zero amplitude for display policy")
    # Contract fixes divide-first float64 order at half-boundaries and avoids
    # overflowing 255*A for large finite amplitudes.
    expected = np.floor(255 * (values / maxima) + 0.5).astype(np.uint8)
    actual = conversion(raw)
    if not isinstance(actual, np.ndarray) or actual.dtype != np.uint8 or actual.shape != values.shape:
        raise ValueError("amplitude_to_u8 must return uint8 with matching 5D shape")
    if not np.array_equal(actual, expected):
        raise ValueError("PNG conversion violates the fixed evaluation policy")
    return actual, maxima.reshape(4).tolist(), torch.from_numpy(values / maxima)


def _metrics(value: object) -> list[dict]:
    if not isinstance(value, list) or len(value) != 1 or len(value[0]) != 4:
        raise ValueError("metrics must be nested [batch=1][four planes]")
    result = [asdict(x) if is_dataclass(x) else x for x in value[0]]
    for pair in result:
        if set(pair) != {"pearson", "ssim", "pearson_defined"}:
            raise ValueError("unexpected per-pair metric schema")
        if type(pair["pearson_defined"]) is not bool:
            raise ValueError("pearson_defined must be bool")
        if pair["pearson_defined"] != (pair["pearson"] is not None):
            raise ValueError("undefined Pearson must be null with explicit status")
        for name in ("pearson", "ssim"):
            value = pair[name]
            if value is None and name == "pearson":
                continue
            if not isinstance(value, float) or not math.isfinite(value) or not -1.00000001 <= value <= 1.00000001:
                raise ValueError(f"invalid {name}")
    json.dumps(result, allow_nan=False)
    return result


def export_run(target_paths: list[Path], phase_rad: torch.Tensor, output_dir: Path, *,
               components: ExportComponents | None = None,
               config: CaseConfigV1 | None = None, metadata: dict | None = None,
               synthetic_test: bool = False) -> dict:
    started = time.perf_counter()
    components = components or production_components()
    config = config or CaseConfigV1()
    if len(target_paths) != 4:
        raise ValueError("exactly four ordered target PNG paths are required")
    targets = _quartet(components.load_quartet(target_paths))
    if not isinstance(phase_rad, torch.Tensor) or phase_rad.shape != (1, 1, 128, 128):
        raise ValueError("one shared phase must have shape [1,1,128,128]")
    metadata = dict(metadata or {})
    json.dumps(metadata, allow_nan=False)  # Reject invalid solver metadata before writing.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier run.
    inputs_dir = output_dir / "targets"
    inputs_dir.mkdir()
    copied_targets = []
    for index, source in enumerate(target_paths, 1):
        copied = inputs_dir / f"target_{index:02d}.png"
        shutil.copyfile(source, copied)
        copied_targets.append(copied)
    doe_path = save_doe_png(phase_rad, output_dir / "DOE.png")
    decoded = load_doe_png(doe_path, dtype=phase_rad.dtype, device=phase_rad.device)
    with torch.no_grad():
        quantized_observables = _reconstruct(decoded, config, components.propagate)
        continuous_observables = _reconstruct(phase_rad.detach(), config, components.propagate)
        quantized_raw = quantized_observables.amplitude
        continuous_raw = continuous_observables.amplitude
        quantized_u8, maxima, quantized_float = _display(quantized_raw, components.amplitude_to_u8)
        continuous_u8, _, continuous_float = _display(continuous_raw, components.amplitude_to_u8)
    reconstruction_paths = [Path(p) for p in components.save_reconstructions(quantized_u8, output_dir)]
    if len(reconstruction_paths) != 4 or len(set(p.resolve() for p in reconstruction_paths)) != 4:
        raise ValueError("save_reconstructions must return four distinct ordered PNG paths")
    for path in reconstruction_paths:
        if path.resolve().parent != output_dir.resolve() or path.resolve() == doe_path.resolve():
            raise ValueError("reconstruction path must be a separate file in this run directory")
    reloaded = _quartet(components.load_quartet(reconstruction_paths))
    actual_codes = torch.round(reloaded * 255).to(torch.uint8).cpu().numpy()
    if not np.array_equal(actual_codes, quantized_u8):
        raise ValueError("saved reconstruction pixels differ from decoded-DOE reconstruction")
    primary = _metrics(components.evaluate(reloaded, targets))
    continuous_display = torch.from_numpy(continuous_u8.astype(np.float64) / 255)
    continuous_metrics = _metrics(components.evaluate(continuous_display, targets.to(torch.float64)))
    quantized_float_metrics = _metrics(components.evaluate(quantized_float, targets.to(torch.float64)))
    continuous_float_metrics = _metrics(components.evaluate(continuous_float, targets.to(torch.float64)))
    delta = [{name: (None if q[name] is None or c[name] is None else q[name] - c[name])
              for name in ("pearson", "ssim")}
             for q, c in zip(quantized_float_metrics, continuous_float_metrics)]
    doe_hash = sha256(doe_path)
    metrics = {"primary_source": "reloaded_uint8_PNG_from_reloaded_DOE",
               "observation": OBSERVATION_V1, "contract": CONTRACT_VERSION_V1,
               "policy": RECONSTRUCTION_POLICY_V1, "ssim_policy": SSIM_POLICY_V1,
               "planes": [{"index": i, "distance_m": config.distances_m[i],
                           "doe_sha256": doe_hash, "reconstruction": reconstruction_paths[i].name,
                           **primary[i]} for i in range(4)],
               "continuous_phase_display_metrics": continuous_metrics,
               "decoded_phase_float_display_metrics": quantized_float_metrics,
               "continuous_phase_float_display_metrics": continuous_float_metrics,
               "quantization_delta": {"definition": "decoded minus continuous; float display domain",
                                      "planes": delta}}
    _write_json(output_dir / "metrics.json", metrics)
    np.savez_compressed(output_dir / "raw.npz", phase_continuous=phase_rad.detach().cpu().numpy(),
                        phase_decoded=decoded.cpu().numpy(), amplitude_quantized=quantized_raw.cpu().numpy(),
                        amplitude_continuous=continuous_raw.cpu().numpy(),
                        intensity_quantized=quantized_observables.intensity.cpu().numpy(),
                        intensity_continuous=continuous_observables.intensity.cpu().numpy())
    files = [*copied_targets, doe_path, *reconstruction_paths,
             output_dir / "metrics.json", output_dir / "raw.npz"]
    manifest = {"schema_version": 2, "case_contract": CONTRACT_VERSION_V1, "observation": OBSERVATION_V1,
                "completed": True,
                "synthetic_test": bool(synthetic_test), "config": asdict(config),
                "output_pitches_m": [config.plane(i).output_pitch_m(128, 128) for i in range(4)],
                "reconstruction_policy": RECONSTRUCTION_POLICY_V1, "ssim_policy": SSIM_POLICY_V1,
                "plane_maxima": maxima, "plane_exposures": [1 / x for x in maxima],
                "git_sha": _git_sha(), "python": platform.python_version(),
                "dependencies": {name: version(name) for name in ("torch", "numpy", "Pillow")},
                "input_sources": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in target_paths],
                "files": {str(p.relative_to(output_dir)): sha256(p) for p in files},
                "metadata": metadata, "elapsed_export_s": time.perf_counter() - started,
                "timing_scope": "DOE write/reload, 4 primary + 4 continuous forwards, PNG/metric/raw export"}
    _write_json(output_dir / "manifest.json", manifest)  # Written last: absent on failure.
    return manifest
