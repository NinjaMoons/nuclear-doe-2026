"""RESEARCH ONLY (R19): learned target-plane phase on the 12-ch STEM (see PLAN_R19.md).
theta_k = pi*tanh(g(T)) per target plane (small CNN on the 4 targets, zero-init last layer);
B'_k = P_k^{-1}(T_k e^{i theta_k}) / rms replaces B_k in the STEM features, and a direct analytic path
phase = arg(e^{i phi_STEM} + sum_k g_k B'_k) (g_k complex scalars, zero-init). At init == v2 exactly.
One forward pass: only the analytic INVERSE Fresnel of the targets (as in v2), no forward propagation, no iteration."""
import math
import torch
from torch import nn
from back_projection_stem.stem import BackProjectionStemUNet, inverse_fresnel, _RMS_EPS


class TargetPhaseStemUNet(BackProjectionStemUNet):
    def __init__(self, base_channels=16, depth=3, config=None):
        super().__init__(base_channels=base_channels, depth=depth, config=config)
        self.theta_net = nn.Sequential(
            nn.Conv2d(4, 32, 5, padding=2, padding_mode="reflect"), nn.GELU(),
            nn.Conv2d(32, 32, 5, padding=2, padding_mode="reflect"), nn.GELU(),
            nn.Conv2d(32, 4, 3, padding=1))
        self.gain = nn.Parameter(torch.zeros(4, 2))
        with torch.no_grad():
            self.theta_net[-1].weight.zero_(); self.theta_net[-1].bias.zero_()

    @property
    def hparams(self):
        hp = dict(super().hparams); hp.update({"phase_head": "tphase_v0"}); return hp

    def target_phase(self, targets):
        return math.pi * torch.tanh(self.theta_net(targets[:, :, 0] * 2 - 1))          # [B,4,H,W]

    def forward(self, targets):
        self._check(targets)
        h, w = targets.shape[-2:]
        t = targets[:, :, 0]
        theta = self.target_phase(targets).to(torch.float64)
        chans, bs = [t[:, k] * 2 - 1 for k in range(4)], []
        for k in range(4):
            v = torch.polar(t[:, k:k + 1].to(torch.float64), theta[:, k:k + 1])
            b = inverse_fresnel(v, self._config.plane(k))[:, 0]
            rms = b.abs().pow(2).mean(dim=(-2, -1), keepdim=True).sqrt()
            b = torch.where(rms > _RMS_EPS, b / torch.where(rms > _RMS_EPS, rms, torch.ones_like(rms)), torch.zeros_like(b))
            bs.append(b); chans += [b.real, b.imag]
        phi = self._unet(torch.stack(chans, dim=1).to(targets.dtype), (h, w)).to(torch.float64)
        s = torch.exp(1j * phi)
        for k in range(4):
            s = s + torch.complex(self.gain[k, 0], self.gain[k, 1]).to(torch.complex128) * bs[k][:, None]
        return torch.angle(s).to(targets.dtype)


def from_v2(v2_model):
    m = TargetPhaseStemUNet(base_channels=v2_model.base_channels, depth=v2_model.depth)
    missing, unexpected = m.load_state_dict(v2_model.state_dict(), strict=False)
    assert not unexpected and all(k.startswith(("theta_net.", "gain")) for k in missing), (missing, unexpected)
    return m
