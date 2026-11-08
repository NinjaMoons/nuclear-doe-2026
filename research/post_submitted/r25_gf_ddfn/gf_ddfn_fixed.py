"""R25 (NIH-POST-R25-GF-DDFN-037) EXPERIMENTAL WORKING COPY of Gemini's GF-DDFN with the physics/codec bugs fixed.
The archived original ../gemini_alexander/gf_ddfn_GEMINI_UNTESTED.py is imported read-only and never modified.

Fixes (each verified in physics_gate.py / test_r25.py):
  1. STE 8-bit quantizer: original round(255 q)/256 -> production codec exactly:
     code = floor(255 (phi+pi)/(2 pi) + 0.5), phi_q = -pi + 2 pi code / 255  (doe_experiments.phase_codec).
  2. Adjoint lift: original omits the complex per-plane scale s = exp(i 2 pi z/lam)/(i lam z) sqrt(HW) dy dx of the
     production single-FFT forward, so it equals F^H / conj(s) (|s| = 0.998..0.869 over the 4 planes and a different
     constant phase per plane). Fixed: exact discrete adjoint F^H built from the production factors
     (back_projection_stem.stem._factors). lift="inverse" gives the physical (pixel-area weighted) adjoint, which equals
     F^-1 because F is a physical isometry.
  3. Router: deterministic softmax gates in both train and eval (no Gumbel noise, no hard top-1 switch) so that training
     and inference compute the same function. (Not used by the R25 one-factor pilot at all.)
Chirps, grids, fftshift order and FFT sign of the original were checked and are correct.
"""
import importlib.util
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from back_projection_stem.stem import _factors, inverse_fresnel
from doe_core.contracts_v1 import CaseConfigV1

RAW_PATH = Path(__file__).resolve().parents[1] / "gemini_alexander" / "gf_ddfn_GEMINI_UNTESTED.py"
_spec = importlib.util.spec_from_file_location("gf_ddfn_gemini_raw", RAW_PATH)
raw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(raw)


class STEQuantizeL8(torch.autograd.Function):
    """Production L8 DOE codec as a straight-through op (forward bit-exact with encode_phase/decode_phase)."""

    @staticmethod
    def forward(ctx, phase):
        # values already in [-pi, pi] are encoded as-is (official: +pi -> code 255); only out-of-range values are wrapped
        p = torch.where(phase.abs() <= math.pi, phase, torch.remainder(phase + math.pi, 2 * math.pi) - math.pi)
        code = torch.floor(255.0 * (p + math.pi) / (2 * math.pi) + 0.5).clamp(0, 255)
        return -math.pi + (2 * math.pi / 255.0) * code

    @staticmethod
    def backward(ctx, g):
        return g


def adjoint_fresnel(V, geometry):
    """Exact discrete adjoint F^H of production propagate_fresnel (plain complex inner product). V: [B,1,H,W] complex."""
    h, w = V.shape[-2:]
    c_in, o = _factors(h, w, geometry)
    to = dict(device=V.device, dtype=V.dtype)
    x = torch.fft.ifftshift(V * o.conj().to(**to), dim=(-2, -1))
    x = torch.fft.fftshift(torch.fft.ifft2(x, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
    return x * c_in.conj().to(**to)


class GF_DDFN_Fixed(raw.GF_DDFN):
    def __init__(self, *a, lift="adjoint", **k):
        super().__init__(*a, **k)
        assert lift in ("adjoint", "inverse")
        self.lift, self.cfg = lift, CaseConfigV1()

    def adjoint_fresnel_lift(self, U_target):
        op = adjoint_fresnel if self.lift == "adjoint" else inverse_fresnel
        return torch.cat([op(U_target[:, k:k + 1], self.cfg.plane(k)) for k in range(U_target.shape[1])], dim=1)

    def forward(self, T):
        dA, dPsi = torch.chunk(self.target_head(self.target_encoder(T)), 2, dim=1)
        A_comp = F.relu(T + 0.3 * torch.tanh(dA)) + 1e-4
        U = self.adjoint_fresnel_lift(torch.polar(A_comp, self.psi_anchor + math.pi * torch.tanh(dPsi)))
        S = U.mean(dim=1, keepdim=True)
        S = S / S.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        f = self.doe_trunk(torch.cat([U.real, U.imag, S.real, S.imag], dim=1).to(T.dtype))
        gates = torch.softmax(self.router(self.compute_regime_features(T)), dim=-1).view(-1, 2, 1, 1, 1).to(T.dtype)
        res = gates[:, 0] * self.head_expert_A(f) + gates[:, 1] * self.head_expert_B(f)
        tot = torch.cat([S.real, S.imag], dim=1).to(T.dtype) + res
        return STEQuantizeL8.apply(torch.atan2(tot[:, 1:2], tot[:, 0:1]))
