"""R25 ONE-FACTOR model (NIH-POST-R25-GF-DDFN-037): STEM V2 + GF-DDFN target-conditioned target-plane phase dPsi_k(T).

Only change vs STEM V2: the zero target phase of the stem lift V_k = T_k + 0j becomes V_k = T_k exp(i dPsi_k(T)),
dPsi = pi tanh(head(enc(2T-1))); enc = GF-DDFN `target_encoder` (Conv 4->32, SiLU, 3 ResBlocks with GroupNorm),
head = 1x1 conv with ONLY the 4 dPsi channels (no amplitude compensation dA), zero-init -> exactly V2 at init.
Lift = production inverse_fresnel (== physical adjoint, exact; see physics_gate.py), complex128, per-plane rms as V2.
NOT included (separate future factors): phase anchor, 2-expert router, amplitude compensation, DOE-plane coherent-sum
residual head, STE quantizer in training (both arms train continuous phase and are evaluated through the official
L8 DOE/PNG export).
Diff vs R19 tphase: R19 additionally has a direct analytic path phase = arg(e^{i phi} + sum_k g_k B'_k) and a 3-layer 5x5
GELU theta_net; R25 has no direct path and uses the GF-DDFN ResBlock encoder. R22 feeds forward-propagated residual
fields instead (different mechanism).
"""
import math

import torch
from torch import nn

from back_projection_stem.stem import BackProjectionStemUNet, inverse_fresnel, _RMS_EPS


class ResBlock(nn.Module):  # same as the Gemini prototype ResBlock (non-inplace SiLU)
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1, padding_mode="circular"), nn.GroupNorm(8, ch), nn.SiLU(),
                                 nn.Conv2d(ch, ch, 3, padding=1, padding_mode="circular"), nn.GroupNorm(8, ch))

    def forward(self, x):
        return nn.functional.silu(x + self.net(x))


class DPsiStemUNet(BackProjectionStemUNet):
    def __init__(self, base_channels=16, depth=3, config=None, width=32):
        super().__init__(base_channels=base_channels, depth=depth, config=config)
        self.dpsi_encoder = nn.Sequential(nn.Conv2d(4, width, 3, padding=1), nn.SiLU(),
                                          ResBlock(width), ResBlock(width), ResBlock(width))
        self.dpsi_head = nn.Conv2d(width, 4, 1)
        nn.init.zeros_(self.dpsi_head.weight)
        nn.init.zeros_(self.dpsi_head.bias)

    @property
    def hparams(self):
        hp = dict(super().hparams)
        hp.update({"phase_head": "r25_dpsi_v0"})
        return hp

    def target_phase(self, targets):
        return math.pi * torch.tanh(self.dpsi_head(self.dpsi_encoder(targets[:, :, 0] * 2 - 1)))

    def features(self, targets):
        t = targets[:, :, 0]
        psi = self.target_phase(targets).to(torch.float64)
        chans = [t[:, k] * 2 - 1 for k in range(4)]
        for k in range(4):
            v = torch.polar(t[:, k:k + 1].to(torch.float64), psi[:, k:k + 1])
            b = inverse_fresnel(v, self._config.plane(k))[:, 0]
            rms = b.abs().pow(2).mean(dim=(-2, -1), keepdim=True).sqrt()
            b = torch.where(rms > _RMS_EPS, b / torch.where(rms > _RMS_EPS, rms, torch.ones_like(rms)), torch.zeros_like(b))
            chans += [b.real, b.imag]
        return torch.stack(chans, dim=1).to(targets.dtype)

    def forward(self, targets):
        self._check(targets)
        return self._unet(self.features(targets), targets.shape[-2:])


def from_v2(v2_model):
    m = DPsiStemUNet(base_channels=v2_model.base_channels, depth=v2_model.depth)
    missing, unexpected = m.load_state_dict(v2_model.state_dict(), strict=False)
    assert not unexpected and all(k.startswith(("dpsi_encoder.", "dpsi_head.")) for k in missing), (missing, unexpected)
    return m
