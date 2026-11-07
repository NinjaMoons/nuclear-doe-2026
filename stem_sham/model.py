"""RESEARCH ONLY (#44 P1 sham): 12-channel STEM variants with replaced 8 extra channels (see PLAN_SHAM.md)."""
import json, math
from pathlib import Path
import torch
from back_projection_stem.stem import BackProjectionStemUNet, backprojection_features
from doe_core.contracts_v1 import CaseConfigV1

N_BANK = 32


def chirp_features(b, h, w, dtype, device, config=None):
    cfg = config or CaseConfigV1(); chans = []
    for k in range(4):
        g = cfg.plane(k); dy, dx = g.input_pitch_m
        y = (torch.arange(h, dtype=torch.float64) - h // 2) * dy; x = (torch.arange(w, dtype=torch.float64) - w // 2) * dx
        ph = math.pi * (y[:, None] ** 2 + x[None, :] ** 2) / (g.wavelength_m * g.distance_m)
        chans += [torch.cos(ph), torch.sin(ph)]
    return torch.stack(chans)[None].expand(b, -1, -1, -1).to(dtype=dtype, device=device)


def load_bank(manifest):
    from doe_experiments.data import load_png_quartet
    m = Path(manifest); man = json.loads(m.read_text())
    tr = [s for s in man["samples"] if s["split"] == "train"][:N_BANK]
    return torch.cat([load_png_quartet([m.parent / p for p in s["paths"]]) for s in tr]), [s["key"] for s in tr]


class ShamStemUNet(BackProjectionStemUNet):
    def __init__(self, base_channels=16, depth=3, config=None, sham="chirp"):
        super().__init__(base_channels=base_channels, depth=depth, config=config)
        self.sham = sham
        if sham == "shuffle":
            self.register_buffer("bank", torch.zeros(N_BANK, 4, 1, 128, 128))

    @property
    def hparams(self):
        hp = dict(super().hparams); hp.update({"sham": self.sham, "sham_version": "v0"}); return hp

    def donor_index(self, targets):
        codes = torch.round(targets * 255).flatten(1).sum(1).long()
        idx = codes % N_BANK
        for i in range(targets.shape[0]):
            if torch.equal(self.bank[idx[i]].to(targets.dtype), targets[i]):
                idx[i] = (idx[i] + 1) % N_BANK
        return idx

    def forward(self, targets):
        self._check(targets)
        base = backprojection_features(targets, self._config)
        if self.sham == "chirp":
            extra = chirp_features(targets.shape[0], *targets.shape[-2:], base.dtype, base.device, self._config)
        elif self.sham == "shuffle":
            donors = self.bank[self.donor_index(targets)].to(dtype=targets.dtype, device=targets.device)
            extra = backprojection_features(donors, self._config)[:, 4:]
        else:
            raise ValueError(self.sham)
        return self._unet(torch.cat([base[:, :4], extra], dim=1), targets.shape[-2:])


def build_sham_model(*, seed, base_channels=16, depth=3, sham="chirp", bank=None):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        m = ShamStemUNet(base_channels=base_channels, depth=depth, sham=sham)
    if sham == "shuffle":
        with torch.no_grad():
            m.bank.copy_(bank)
    return m
