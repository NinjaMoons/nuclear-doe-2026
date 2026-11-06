"""#4B pilot training corpus: 64 A + 64 B amplitude quartets (contract v1.1-amplitude).

A (achievable): origin phase -> existing 8-bit DOE codec -> saved DOE.png -> RELOAD
-> decoded phase phi_q -> ONE field exp(i*phi_q) -> existing single-FFT Fresnel at the
four official z -> |U| -> existing amplitude_to_u8 -> four PNG. The phase label is
phi_q from DOE.png; the origin phase is provenance only.

B (arbitrary): procedural synthetic quartets (geometry, stroke symbols, lines,
grayscale fields); no DOE and no phase label.

Splits are assigned per origin ID before any rendering. Every file is SHA256-hashed
in manifest.json. Physics, codec, PNG IO and display policy are imported, never
re-implemented here. Data A is reachable for the discrete operator only; this is not
a statement about a physical bench.

CLI:
    python -m doe_experiments.corpus generate --output-dir DIR [--seed 0]
    python -m doe_experiments.corpus validate --manifest DIR/manifest.json
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw
import torch

from doe_core.contracts_v1 import (CONTRACT_VERSION_V1, OBSERVATION_V1, RECONSTRUCTION_POLICY_V1,
                                   CaseConfigV1)
from doe_core.fresnel import propagate_fresnel
from doe_experiments.data import load_png_quartet
from doe_experiments.phase_codec import load_doe_png, save_doe_png
from doe_experiments.png_io import amplitude_to_u8, save_reconstructions
from doe_models.adam_multiplane import multiplane_amplitudes

__all__ = ["SCHEMA", "GENERATOR_VERSION", "CorpusSplit", "generate_corpus", "validate_corpus",
           "load_split", "load_training_splits", "file_sha256", "content_sha256", "main"]

SCHEMA = "doe_training_manifest_v1"
GENERATOR_VERSION = "4b-pilot-1"
SPLITS = ("train", "val", "test")
CORPORA = (("A64", "A"), ("B64", "B"))
DEFAULT_COUNTS = (64, 64)
DEFAULT_SPLIT_SIZES = (48, 8, 8)
SIZE = 128
A_PHASE_FAMILIES = ("lowpass_random", "zernike_mix", "piecewise_levels", "lens_tilt_lowpass")
B_FAMILIES = ("geometry", "symbols", "lines", "grayscale")
_SUPERSAMPLE = 4
_HEX = frozenset("0123456789abcdef")


# --------------------------------------------------------------------------- helpers

def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(samples: Sequence[dict]) -> str:
    """Platform-independent digest of corpus CONTENT: keys, origins, splits and file hashes."""
    rows = [[s.get("key"), s.get("origin_id"), s.get("split"), s.get("kind"), s.get("sha256"),
             s.get("doe_sha256")] for s in samples]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _rng(seed: int, *key: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=tuple(key)))


def _check_seed(seed: object) -> int:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative int")
    return seed


def _git_state() -> dict:
    root = Path(__file__).resolve().parents[1]
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                             text=True, check=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        return {"git_sha": sha, "tracked_tree_dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {"git_sha": None, "tracked_tree_dirty": None}


def _environment() -> dict:
    import PIL
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__, "pillow": PIL.__version__}


def _wrap(phase: np.ndarray) -> np.ndarray:
    """Wrap to [-pi, pi] (float64); +pi is kept as allowed by the codec."""
    return np.clip(np.angle(np.exp(1j * phase)), -math.pi, math.pi)


def _assign_splits(seed: int, corpus_index: int, count: int, sizes: Sequence[int]) -> list[str]:
    """Split labels per origin index, fixed BEFORE any sample is rendered."""
    if len(sizes) != 3 or any(type(s) is not int or s < 1 for s in sizes) or sum(sizes) != count:
        raise ValueError("split sizes must be three positive ints summing to the corpus count")
    order = _rng(seed, corpus_index, 999).permutation(count)
    labels = [""] * count
    position = 0
    for name, size in zip(SPLITS, sizes):
        for index in order[position:position + size]:
            labels[int(index)] = name
        position += size
    return labels


def _grid() -> tuple[np.ndarray, np.ndarray]:
    axis = (np.arange(SIZE, dtype=np.float64) - SIZE // 2) / (SIZE / 2)
    return np.meshgrid(axis, axis, indexing="ij")  # y, x in [-1, 1)


def _lowpass_noise(rng: np.random.Generator, cutoff: float) -> np.ndarray:
    """Real Gaussian random field, unit std, isotropic Gaussian spectrum."""
    white = rng.standard_normal((SIZE, SIZE))
    fy = np.fft.fftfreq(SIZE)[:, None]
    fx = np.fft.fftfreq(SIZE)[None, :]
    field = np.fft.ifft2(np.fft.fft2(white) * np.exp(-(fx**2 + fy**2) / (2 * cutoff**2))).real
    return (field - field.mean()) / max(field.std(), 1e-12)


# --------------------------------------------------------------------------- corpus A

def _origin_phase(rng: np.random.Generator, family: str) -> tuple[np.ndarray, dict]:
    y, x = _grid()
    if family == "lowpass_random":
        # Gentle phase gradients: steeper fields (cutoff>0.03, >3pi) degenerate into speckle.
        cutoff = float(rng.uniform(0.004, 0.02))
        amplitude = float(rng.uniform(1.0, 3.0)) * math.pi
        return _wrap(amplitude * _lowpass_noise(rng, cutoff)), {"cutoff": cutoff, "amplitude_rad": amplitude}
    if family == "zernike_mix":
        r2 = x**2 + y**2
        terms = {"tilt_x": x, "tilt_y": y, "defocus": 2 * r2 - 1, "astig_0": x**2 - y**2,
                 "astig_45": 2 * x * y, "coma_x": (3 * r2 - 2) * x, "coma_y": (3 * r2 - 2) * y,
                 "spherical": 6 * r2**2 - 6 * r2 + 1}
        coeffs = {name: float(rng.normal(0.0, 6.0)) for name in terms}
        phase = sum(coeffs[name] * terms[name] for name in terms)
        return _wrap(phase), {"coefficients_rad": coeffs}
    if family == "piecewise_levels":
        levels = int(rng.integers(2, 9))
        cutoff = float(rng.uniform(0.02, 0.08))
        noise = _lowpass_noise(rng, cutoff)
        edges = np.quantile(noise, np.linspace(0, 1, levels + 1)[1:-1])
        values = rng.uniform(-math.pi, math.pi, levels)
        phase = values[np.digitize(noise, edges)]
        return _wrap(phase), {"levels": levels, "cutoff": cutoff, "level_values_rad": values.tolist()}
    if family == "lens_tilt_lowpass":
        curvature = float(rng.uniform(-40.0, 40.0))
        tilt = rng.uniform(-30.0, 30.0, 2)
        cutoff = float(rng.uniform(0.01, 0.04))
        noise_amp = float(rng.uniform(0.5, 2.0)) * math.pi
        phase = curvature * (x**2 + y**2) + tilt[0] * y + tilt[1] * x + noise_amp * _lowpass_noise(rng, cutoff)
        return _wrap(phase), {"curvature_rad": curvature, "tilt_rad": tilt.tolist(), "cutoff": cutoff,
                              "noise_amplitude_rad": noise_amp}
    raise ValueError(f"unknown A phase family {family!r}")


def _forward_from_doe(doe_path: Path, config: CaseConfigV1) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode the saved DOE and run the shared four-plane forward. Returns (phi_q, raw |U|)."""
    phi_q = load_doe_png(doe_path, dtype=torch.float64)
    return phi_q, multiplane_amplitudes(phi_q, config, propagate_fresnel)


def _make_a(sample_dir: Path, rng: np.random.Generator, family: str, config: CaseConfigV1) -> dict:
    origin, params = _origin_phase(rng, family)
    sample_dir.mkdir(parents=True, exist_ok=False)
    origin_path = sample_dir / "origin_phase.npy"
    with origin_path.open("xb") as stream:
        np.save(stream, origin, allow_pickle=False)
    doe_path = save_doe_png(origin[None, None], sample_dir / "DOE.png")
    phi_q, amplitudes = _forward_from_doe(doe_path, config)
    images = amplitude_to_u8(amplitudes)
    paths = save_reconstructions(images, sample_dir)
    maxima = amplitudes.detach().double().amax(dim=(-2, -1)).reshape(4).tolist()
    quant_error = float(np.abs(_wrap(phi_q[0, 0].numpy() - origin)).max())
    if not quant_error <= math.pi / 255 + 1e-12:
        raise ValueError("decoded DOE deviates from origin phase by more than one codec step")
    return {"paths": paths, "doe_path": doe_path, "origin_path": origin_path, "origin_params": params,
            "raw_amplitude_max": maxima, "quant_error": quant_error}


# --------------------------------------------------------------------------- corpus B

_STROKES = {  # unit-box polylines (x, y), y down
    "0": [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]], "1": [[(0.5, 0), (0.5, 1)], [(0.25, 0.2), (0.5, 0)]],
    "2": [[(0, 0), (1, 0), (1, 0.5), (0, 0.5), (0, 1), (1, 1)]], "3": [[(0, 0), (1, 0), (1, 1), (0, 1)], [(0.2, 0.5), (1, 0.5)]],
    "4": [[(0, 0), (0, 0.5), (1, 0.5)], [(1, 0), (1, 1)]], "5": [[(1, 0), (0, 0), (0, 0.5), (1, 0.5), (1, 1), (0, 1)]],
    "7": [[(0, 0), (1, 0), (0.4, 1)]], "A": [[(0, 1), (0.5, 0), (1, 1)], [(0.25, 0.55), (0.75, 0.55)]],
    "E": [[(1, 0), (0, 0), (0, 1), (1, 1)], [(0, 0.5), (0.7, 0.5)]], "F": [[(1, 0), (0, 0), (0, 1)], [(0, 0.5), (0.7, 0.5)]],
    "H": [[(0, 0), (0, 1)], [(1, 0), (1, 1)], [(0, 0.5), (1, 0.5)]], "K": [[(0, 0), (0, 1)], [(1, 0), (0, 0.55), (1, 1)]],
    "L": [[(0, 0), (0, 1), (1, 1)]], "M": [[(0, 1), (0, 0), (0.5, 0.6), (1, 0), (1, 1)]],
    "N": [[(0, 1), (0, 0), (1, 1), (1, 0)]], "T": [[(0, 0), (1, 0)], [(0.5, 0), (0.5, 1)]],
    "V": [[(0, 0), (0.5, 1), (1, 0)]], "X": [[(0, 0), (1, 1)], [(1, 0), (0, 1)]], "Z": [[(0, 0), (1, 0), (0, 1), (1, 1)]],
    "+": [[(0.5, 0), (0.5, 1)], [(0, 0.5), (1, 0.5)]], "arrow": [[(0, 0.5), (1, 0.5)], [(0.65, 0.2), (1, 0.5), (0.65, 0.8)]],
}
_GLYPHS = tuple(_STROKES)


def _canvas(level: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("L", (SIZE * _SUPERSAMPLE, SIZE * _SUPERSAMPLE), level)
    return image, ImageDraw.Draw(image)


def _downsample(image: Image.Image) -> np.ndarray:
    big = np.asarray(image, dtype=np.float64) / 255.0
    return big.reshape(SIZE, _SUPERSAMPLE, SIZE, _SUPERSAMPLE).mean(axis=(1, 3))


def _render_b(rng: np.random.Generator, family: str) -> tuple[np.ndarray, dict]:
    s = SIZE * _SUPERSAMPLE
    if family == "geometry":
        image, draw = _canvas(int(rng.integers(0, 40)))
        shapes = []
        for _ in range(int(rng.integers(2, 7))):
            kind = str(rng.choice(["ellipse", "rectangle", "polygon"]))
            fill = int(rng.integers(90, 256))
            cx, cy = rng.uniform(0.15, 0.85, 2) * s
            rx, ry = rng.uniform(0.05, 0.25, 2) * s
            if kind == "polygon":
                n = int(rng.integers(3, 8))
                angles = np.sort(rng.uniform(0, 2 * math.pi, n))
                points = [(float(cx + rx * math.cos(a)), float(cy + ry * math.sin(a))) for a in angles]
                draw.polygon(points, fill=fill)
            else:
                box = [float(cx - rx), float(cy - ry), float(cx + rx), float(cy + ry)]
                (draw.ellipse if kind == "ellipse" else draw.rectangle)(box, fill=fill)
            shapes.append(kind)
        return _downsample(image), {"shapes": shapes}
    if family == "symbols":
        image, draw = _canvas(int(rng.integers(0, 30)))
        glyphs = []
        for _ in range(int(rng.integers(1, 4))):
            glyph = _GLYPHS[int(rng.integers(len(_GLYPHS)))]
            scale = float(rng.uniform(0.25, 0.6)) * s
            angle = float(rng.uniform(-0.6, 0.6))
            ox, oy = rng.uniform(0.1, 0.9 - scale / s, 2) * s
            width = int(rng.integers(3, 6)) * _SUPERSAMPLE
            fill = int(rng.integers(150, 256))
            c, si = math.cos(angle), math.sin(angle)
            for stroke in _STROKES[glyph]:
                pts = [(float(ox + scale * ((px - 0.5) * c - (py - 0.5) * si + 0.5)),
                        float(oy + scale * ((px - 0.5) * si + (py - 0.5) * c + 0.5))) for px, py in stroke]
                draw.line(pts, fill=fill, width=width, joint="curve")
            glyphs.append(glyph)
        return _downsample(image), {"glyphs": glyphs}
    if family == "lines":
        image, draw = _canvas(int(rng.integers(0, 40)))
        mode = str(rng.choice(["segments", "grid", "rings"]))
        if mode == "segments":
            for _ in range(int(rng.integers(3, 10))):
                p = rng.uniform(0, s, 4)
                draw.line([float(v) for v in p], fill=int(rng.integers(100, 256)),
                          width=int(rng.integers(1, 5)) * _SUPERSAMPLE)
        elif mode == "grid":
            step = int(rng.integers(10, 33)) * _SUPERSAMPLE
            width = int(rng.integers(1, 4)) * _SUPERSAMPLE
            fill = int(rng.integers(120, 256))
            offset = int(rng.integers(0, step))
            for v in range(offset, s, step):
                draw.line([v, 0, v, s], fill=fill, width=width)
                if rng.random() < 0.7:
                    draw.line([0, v, s, v], fill=fill, width=width)
        else:
            cx, cy = rng.uniform(0.3, 0.7, 2) * s
            step = float(rng.uniform(6, 16)) * _SUPERSAMPLE
            width = int(rng.integers(1, 4)) * _SUPERSAMPLE
            fill = int(rng.integers(120, 256))
            r = step
            while r < s:
                draw.ellipse([float(cx - r), float(cy - r), float(cx + r), float(cy + r)], outline=fill, width=width)
                r += step
        return _downsample(image), {"mode": mode}
    if family == "grayscale":
        y, x = _grid()
        mode = str(rng.choice(["linear_gradient", "radial_gradient", "gaussian_blobs", "lowpass_texture"]))
        if mode == "linear_gradient":
            a = float(rng.uniform(0, 2 * math.pi))
            values = 0.5 + 0.5 * (math.cos(a) * x + math.sin(a) * y) / math.sqrt(2)
        elif mode == "radial_gradient":
            cy, cx = rng.uniform(-0.5, 0.5, 2)
            values = 1.0 - np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / 2.2
        elif mode == "gaussian_blobs":
            values = np.zeros_like(x)
            for _ in range(int(rng.integers(2, 8))):
                cy, cx = rng.uniform(-0.8, 0.8, 2)
                sigma = float(rng.uniform(0.05, 0.35))
                values += rng.uniform(0.3, 1.0) * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
        else:
            values = _lowpass_noise(rng, float(rng.uniform(0.02, 0.1)))
        values = values - values.min()
        values = values / max(values.max(), 1e-12) * float(rng.uniform(0.6, 1.0))
        return np.clip(values, 0.0, 1.0), {"mode": mode}
    raise ValueError(f"unknown B family {family!r}")


def _to_u8(values: np.ndarray) -> np.ndarray:
    return np.floor(255 * np.clip(values, 0.0, 1.0) + 0.5).astype(np.uint8)


def _make_b(sample_dir: Path, rng: np.random.Generator) -> dict:
    planes, families, params = [], [], []
    for _ in range(4):
        for _attempt in range(16):  # deterministic redraw of degenerate (near-constant/dark) planes
            family = B_FAMILIES[int(rng.integers(len(B_FAMILIES)))]
            values, info = _render_b(rng, family)
            u8 = _to_u8(values)
            if int(u8.max()) >= 64 and float(u8.std()) >= 4.0:
                break
        else:
            raise ValueError("could not render a non-degenerate B plane")
        planes.append(u8)
        families.append(family)
        params.append(info)
    sample_dir.mkdir(parents=True, exist_ok=False)
    paths = save_reconstructions(np.stack(planes)[None, :, None], sample_dir)
    return {"paths": paths, "families": families, "params": params}


# --------------------------------------------------------------------------- generation

def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def generate_corpus(output_dir: str | Path, *, seed: int = 0, counts: Sequence[int] = DEFAULT_COUNTS,
                    split_sizes: Sequence[int] = DEFAULT_SPLIT_SIZES) -> Path:
    """Generate the pilot into a NEW/empty directory and return the manifest path."""
    seed = _check_seed(seed)
    if len(counts) != 2 or any(type(c) is not int or c < 3 for c in counts):
        raise ValueError("counts must be two ints >= 3 (A, B)")
    root = Path(output_dir)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("output directory must not exist or must be empty (no overwrite)")
    root.mkdir(parents=True, exist_ok=True)
    config = CaseConfigV1()
    samples = []
    for corpus_index, ((corpus_id, kind), count) in enumerate(zip(CORPORA, counts)):
        sizes = tuple(split_sizes) if count == sum(split_sizes) else None
        if sizes is None:
            raise ValueError(f"split sizes {tuple(split_sizes)} do not sum to {corpus_id} count {count}")
        labels = _assign_splits(seed, corpus_index, count, sizes)
        for index in range(count):
            local = f"{index:03d}"
            rng = _rng(seed, corpus_index, index)
            sample_dir = root / corpus_id / local
            entry = {"corpus_id": corpus_id, "local_sample_id": local, "key": f"{corpus_id}:{local}",
                     "origin_id": f"{corpus_id}:origin-{local}", "split": labels[index], "kind": kind,
                     "synthetic": True, "seed_spawn_key": [seed, corpus_index, index]}
            if kind == "A":
                family = A_PHASE_FAMILIES[index % len(A_PHASE_FAMILIES)]
                made = _make_a(sample_dir, rng, family, config)
                entry.update({
                    "doe_path": _rel(made["doe_path"], root), "doe_sha256": file_sha256(made["doe_path"]),
                    "phase_label": {"source": "decoded DOE.png (phi_q)", "path": _rel(made["doe_path"], root),
                                    "codec": "doe_experiments.phase_codec v1 (q/255)"},
                    "origin_phase": {"role": "provenance_only", "family": family, "params": made["origin_params"],
                                     "path": _rel(made["origin_path"], root),
                                     "sha256": file_sha256(made["origin_path"]),
                                     "max_abs_wrapped_decode_error_rad": made["quant_error"]},
                    "raw_amplitude_max": made["raw_amplitude_max"],
                    "exposure": [1.0 / m for m in made["raw_amplitude_max"]]})
            else:
                made = _make_b(sample_dir, rng)
                entry.update({"phase_label": None, "doe_path": None,
                              "procedural": {"families": made["families"], "params": made["params"]}})
            entry["paths"] = [_rel(p, root) for p in made["paths"]]
            entry["sha256"] = [file_sha256(p) for p in made["paths"]]
            entry["quartet_sha256"] = hashlib.sha256("".join(entry["sha256"]).encode()).hexdigest()
            samples.append(entry)
    manifest = {
        "schema": SCHEMA, "contract_version": CONTRACT_VERSION_V1, "observation": OBSERVATION_V1,
        "status": "DATA_CANDIDATE", "generator": {"module": "doe_experiments.corpus", "version": GENERATOR_VERSION,
                                                  **_git_state(),
                                                  "command": f"python -m doe_experiments.corpus generate --output-dir <DIR> --seed {seed}"},
        "seed": seed, "environment": _environment(),
        "case_config": {"wavelength_m": config.wavelength_m, "input_pitch_m": list(config.input_pitch_m),
                        "distances_m": list(config.distances_m), "image_shape": list(config.image_shape),
                        "output_pitch_m": [list(config.plane(i).output_pitch_m(SIZE, SIZE)) for i in range(4)],
                        "illumination": "uniform A=1, U=exp(i*phi_q)"},
        "plane_order": ["z1", "z2", "z3", "z4"],
        "reconstruction_policy": RECONSTRUCTION_POLICY_V1,
        "io": {"png_writer": "doe_experiments.png_io.save_reconstructions (file names reconstructed_01..04.png)",
               "png_loader": "doe_experiments.data.load_png_quartet",
               "a_forward": "doe_models.adam_multiplane.multiplane_amplitudes + doe_core.fresnel.propagate_fresnel"},
        "corpora": {cid: {"kind": kind, "count": count,
                          "split_counts": {s: sum(1 for e in samples if e["corpus_id"] == cid and e["split"] == s)
                                           for s in SPLITS}}
                    for (cid, kind), count in zip(CORPORA, counts)},
        "splits": {s: [e["key"] for e in samples if e["split"] == s] for s in SPLITS},
        "notes": ["A is reachable by the discrete single-FFT operator; not evidence about a physical bench.",
                  "B has no phase ground truth; do not train phase-supervised on B.",
                  "Split was assigned per origin_id before rendering; every origin has exactly one sample.",
                  "Byte-identical regeneration is verified on one platform; compare other machines via sha256."],
        "content_sha256": content_sha256(samples),
        "samples": samples,
    }
    manifest_path = root / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=1, sort_keys=False, allow_nan=False)
        stream.write("\n")
    validate_corpus(manifest_path)
    return manifest_path


# --------------------------------------------------------------------------- validation / loading

def _read_manifest(manifest_path: str | Path) -> tuple[dict, Path]:
    manifest_path = Path(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("cannot read corpus manifest") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError(f"manifest schema must be {SCHEMA}")
    if manifest.get("contract_version") != CONTRACT_VERSION_V1:
        raise ValueError("manifest contract_version does not match v1.1-amplitude")
    if not isinstance(manifest.get("samples"), list) or not manifest["samples"]:
        raise ValueError("manifest has no samples")
    return manifest, manifest_path.resolve().parent


def _resolve(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("manifest paths must be non-empty POSIX relative strings")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"manifest path escapes the corpus root: {relative!r}")
    path = (root / Path(*pure.parts)).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"manifest path escapes the corpus root: {relative!r}")
    return path


def _verified(root: Path, relative: object, expected: object) -> Path:
    if not isinstance(expected, str) or len(expected) != 64 or not set(expected) <= _HEX:
        raise ValueError("sha256 entries must be 64 lowercase hex characters")
    path = _resolve(root, relative)
    if not path.is_file():
        raise ValueError(f"missing corpus file {relative!r}")
    if file_sha256(path) != expected:
        raise ValueError(f"sha256 mismatch for {relative!r}")
    return path


def validate_corpus(manifest_path: str | Path, *, check_a_regeneration: bool = True) -> dict:
    """Check structure, hashes, splits/leakage, duplicates, PNG format and A DOE consistency.

    A consistency re-runs the SAME shared forward from the saved DOE.png; it proves the stored
    PNGs come from that DOE, not the physical correctness of the operator (separate gate).
    """
    manifest, root = _read_manifest(manifest_path)
    samples = manifest["samples"]
    keys, origin_split, quartets, file_owner = set(), {}, {}, {}
    config = CaseConfigV1()
    counts: dict[str, dict[str, int]] = {}
    for entry in samples:
        if not isinstance(entry, dict):
            raise ValueError("each sample must be an object")
        cid, local, kind, split = (entry.get(k) for k in ("corpus_id", "local_sample_id", "kind", "split"))
        key = f"{cid}:{local}"
        if not isinstance(cid, str) or not isinstance(local, str) or entry.get("key") != key:
            raise ValueError("sample key must be namespaced corpus_id:local_sample_id")
        if key in keys:
            raise ValueError(f"duplicate sample key {key}")
        keys.add(key)
        if kind not in ("A", "B") or split not in SPLITS:
            raise ValueError(f"{key}: invalid kind/split")
        origin = entry.get("origin_id")
        if not isinstance(origin, str) or not origin:
            raise ValueError(f"{key}: missing origin_id")
        if origin_split.setdefault(origin, split) != split:
            raise ValueError(f"split leakage: origin {origin} appears in several splits")
        paths, hashes = entry.get("paths"), entry.get("sha256")
        if not isinstance(paths, list) or not isinstance(hashes, list) or len(paths) != 4 or len(hashes) != 4:
            raise ValueError(f"{key}: exactly four plane paths and hashes are required")
        files = [_verified(root, p, h) for p, h in zip(paths, hashes)]
        quartet = hashlib.sha256("".join(hashes).encode()).hexdigest()
        if entry.get("quartet_sha256") != quartet:
            raise ValueError(f"{key}: quartet_sha256 mismatch")
        if quartet in quartets:
            raise ValueError(f"duplicate quartet: {key} == {quartets[quartet]}")
        quartets[quartet] = key
        for digest in hashes:
            other = file_owner.setdefault(digest, origin)
            if other != origin and origin_split[other] != split:
                raise ValueError(f"split leakage: identical PNG shared by {origin} and {other}")
        targets = load_png_quartet(files)  # strict 8-bit L 128x128 check by the #4A loader
        if kind == "A":
            label = entry.get("phase_label")
            if not isinstance(label, dict) or label.get("path") != entry.get("doe_path"):
                raise ValueError(f"{key}: A phase label must reference the decoded DOE.png")
            doe = _verified(root, entry.get("doe_path"), entry.get("doe_sha256"))
            origin_info = entry.get("origin_phase") or {}
            if origin_info.get("role") != "provenance_only":
                raise ValueError(f"{key}: origin phase must be provenance only")
            _verified(root, origin_info.get("path"), origin_info.get("sha256"))
            if check_a_regeneration:
                _, amplitudes = _forward_from_doe(doe, config)
                if not np.array_equal(amplitude_to_u8(amplitudes),
                                      np.rint(targets.double().numpy() * 255).astype(np.uint8)):
                    raise ValueError(f"{key}: A PNGs do not match the forward of its saved DOE.png")
        else:
            if entry.get("phase_label") is not None or entry.get("doe_path") is not None:
                raise ValueError(f"{key}: B samples must not carry a phase label or DOE")
            if any(float(targets[0, i].std()) == 0.0 for i in range(4)):
                raise ValueError(f"{key}: B plane is constant")
        counts.setdefault(cid, {s: 0 for s in SPLITS})[split] += 1
    expected_splits = {s: [e["key"] for e in samples if e["split"] == s] for s in SPLITS}
    if manifest.get("splits") != expected_splits:
        raise ValueError("manifest split lists disagree with per-sample splits")
    content = content_sha256(samples)
    if manifest.get("content_sha256") != content:
        raise ValueError("manifest content_sha256 does not match its samples")
    return {"samples": len(samples), "counts": counts, "content_sha256": content,
            "manifest_sha256": file_sha256(Path(manifest_path))}


@dataclass(frozen=True)
class CorpusSplit:
    """Trainer input: targets CPU float32 [N,4,1,128,128] in [0,1] plus aligned IDs."""
    targets: torch.Tensor
    keys: list[str]
    origin_ids: list[str]
    kinds: list[str]


def load_split(manifest_path: str | Path, split: str, *, kinds: Sequence[str] = ("A", "B"),
               verify_hashes: bool = True) -> CorpusSplit:
    """Load one split in manifest order through the #4A PNG loader (no extra normalisation)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if isinstance(kinds, str) or not set(kinds) <= {"A", "B"} or not kinds:
        raise ValueError("kinds must be a non-empty subset of ('A', 'B')")
    manifest, root = _read_manifest(manifest_path)
    tensors, keys, origins, sample_kinds = [], [], [], []
    for entry in manifest["samples"]:
        if entry.get("split") != split or entry.get("kind") not in kinds:
            continue
        if verify_hashes:
            files = [_verified(root, p, h) for p, h in zip(entry["paths"], entry["sha256"])]
        else:
            files = [_resolve(root, p) for p in entry["paths"]]
        tensors.append(load_png_quartet(files))
        keys.append(entry["key"])
        origins.append(entry["origin_id"])
        sample_kinds.append(entry["kind"])
    if not tensors:
        raise ValueError(f"no samples for split={split} kinds={tuple(kinds)}")
    return CorpusSplit(torch.cat(tensors), keys, origins, sample_kinds)


def load_training_splits(manifest_path: str | Path, *, kinds: Sequence[str] = ("A", "B"),
                         verify_hashes: bool = True) -> dict[str, CorpusSplit]:
    """All three splits; raises if any origin_id appears in two splits."""
    splits = {s: load_split(manifest_path, s, kinds=kinds, verify_hashes=verify_hashes) for s in SPLITS}
    seen: dict[str, str] = {}
    for name, part in splits.items():
        for origin in part.origin_ids:
            if seen.setdefault(origin, name) != name:
                raise ValueError(f"split leakage: origin {origin}")
    return splits


# --------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m doe_experiments.corpus", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="generate the 64A+64B pilot into a new directory")
    gen.add_argument("--output-dir", type=Path, required=True)
    gen.add_argument("--seed", type=int, default=0)
    val = sub.add_parser("validate", help="validate an existing corpus manifest")
    val.add_argument("--manifest", type=Path, required=True)
    val.add_argument("--skip-a-regeneration", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            manifest_path = generate_corpus(args.output_dir, seed=args.seed)
            report = validate_corpus(manifest_path, check_a_regeneration=False)
            report["manifest"] = str(manifest_path)
        else:
            report = validate_corpus(args.manifest, check_a_regeneration=not args.skip_a_regeneration)
            report["manifest"] = str(args.manifest)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
