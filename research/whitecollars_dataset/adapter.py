"""WhiteCollars TRAIN adapter (#41). RESEARCH ONLY; the production loader is not modified.

WhiteCollarsTrain(archive, mapping_json, *, role="train") reads the ONE immutable original archive in memory,
checks its sha256 against WC_MAPPING.json (from wc_audit.py) and every member's file sha256, and returns targets as
float32 tensors [B,4,1,128,128] = PNG uint8 / 255 (amplitude, z1..z4 in mapping order). No sqrt, gamma,
per-image max normalisation, resampling, z reordering, intensity conversion or phase labels. role="holdout" exposes
only the predeclared internal holdout (for descriptive diagnostics); non-train splits are never readable.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import external_dataset_audit as base  # noqa: E402

PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def decode_l8(data: bytes, name: str) -> np.ndarray:
    if data[:8] != PNG_SIG or data[12:16] != b"IHDR" or data[24] != 8 or data[25] != 0:
        raise ValueError(f"{name}: not an 8-bit grayscale PNG")
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        if im.mode != "L" or im.size != (128, 128):
            raise ValueError(f"{name}: mode={im.mode} size={im.size}, need L 128x128")
        return np.asarray(im, dtype=np.uint8).copy()


class WhiteCollarsTrain:
    def __init__(self, archive: str | Path, mapping_json: str | Path, *, role: str = "train",
                 expect_archive_sha256: str | None = None):
        if role not in ("train", "holdout"):
            raise ValueError("role must be 'train' or 'holdout'; other splits are not exposed")
        archive = Path(archive)
        self.mapping = json.loads(Path(mapping_json).read_text(encoding="utf-8"))
        if self.mapping.get("schema") != "wc-mapping/1":
            raise ValueError("unexpected mapping schema")
        actual = _sha(archive.read_bytes())
        for expected in (self.mapping["archive_sha256"], expect_archive_sha256):
            if expected and actual != expected.lower():
                raise ValueError(f"archive sha256 {actual} != expected {expected}")
        self.archive_sha256 = actual
        keys = self.mapping["train_keys"] if role == "train" else self.mapping["holdout_keys"]
        if role == "train" and set(keys) & set(self.mapping["holdout_keys"]):
            raise ValueError("holdout keys leaked into train keys")
        for k in keys:
            if self.mapping["quartets"][k]["split"] != "train":
                raise ValueError(f"{k}: only TRAIN-split quartets may be used")
        self.keys = list(keys)
        wanted = {p: (k, i) for k in self.keys for i, p in enumerate(self.mapping["quartets"][k]["paths"])}
        findings = {"unsafe": []}
        arrays: dict[str, list] = {k: [None] * 4 for k in self.keys}
        for name, data in base.members(archive, findings):
            if name not in wanted:
                continue
            k, i = wanted[name]
            if _sha(data) != self.mapping["quartets"][k]["file_sha256"][i]:
                raise ValueError(f"{name}: file sha256 differs from WC_MAPPING")
            arrays[k][i] = decode_l8(data, name)
        missing = [k for k, v in arrays.items() if any(a is None for a in v)]
        if missing:
            raise ValueError(f"{len(missing)} quartets incomplete in archive, e.g. {missing[:3]}")
        self._u8 = np.stack([np.stack(arrays[k]) for k in self.keys])[:, :, None]  # [N,4,1,128,128] uint8

    def __len__(self) -> int:
        return len(self.keys)

    def targets(self, indices=None) -> torch.Tensor:
        """float32 [B,4,1,128,128] in [0,1] = uint8/255; same contract as doe_experiments.data.load_png_quartet."""
        u8 = self._u8 if indices is None else self._u8[np.asarray(indices)]
        return torch.from_numpy(u8.astype(np.float32) / np.float32(255.0))
