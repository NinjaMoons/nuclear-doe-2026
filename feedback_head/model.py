"""RESEARCH ONLY (R22, #51/6714825411): learned optical-feedback two-stage network on the V2 STEM.

Stage 0: the V2 12-ch STEM (weights cloned from 19b05dad; the source file is never modified) -> phi0.
Feedback features (deterministic, differentiable, ONE fixed pass, no iteration, no optimiser, no gradient step):
  U_k = Fresnel_k(exp(i phi0)) at the official z_k / native grid (doe_core.fresnel.propagate_fresnel),
  Ahat_k = |U_k| / max|U_k|  (display policy), r_k = T_k - Ahat_k,
  E_k = Fresnel_k^-1( r_k * exp(i arg U_k) ) / rms   (error field back-projected with the predicted wavefront phase).
Stage 1: a trainable U-Net on 22 channels [sin phi0, cos phi0, T(4), Ahat(4), r(4), Re/Im E(8)] -> bounded delta
  = pi * tanh(head), head zero-init -> phi = wrap(phi0 + delta) == V2 exactly at step 0.
The final DOE is produced by trained parameters (delta), NOT by phi0 + alpha * analytic gradient.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from back_projection_stem.stem import BackProjectionStemUNet, inverse_fresnel, _RMS_EPS
from doe_core.fresnel import propagate_fresnel
from doe_models.unet_multiplane import FourInputUNet, _block, _wrap

FEEDBACK_CHANNELS = 22


class FeedbackStemUNet(BackProjectionStemUNet):
    """BackProjectionStemUNet (stage 0, V2 weights) + `corr` stage-1 U-Net with a zero-init bounded delta head."""

    def __init__(self, base_channels=16, depth=3, config=None, corr_channels=16):
        super().__init__(base_channels=base_channels, depth=depth, config=config)
        corr = FourInputUNet(base_channels=corr_channels, depth=depth)
        corr.encoders[0] = _block(FEEDBACK_CHANNELS, corr.encoders[0][0].out_channels)
        with torch.no_grad():
            corr.head.weight.zero_(); corr.head.bias.zero_()
        self.corr = corr
        self.corr_channels = corr_channels

    @property
    def hparams(self):
        hp = dict(super().hparams)
        hp.update({"feedback_head": "r22_v0", "corr_channels": self.corr_channels, "feedback_channels": FEEDBACK_CHANNELS})
        return hp

    def stage0(self, targets):
        return super().forward(targets)

    def feedback_features(self, targets, phi0):
        t = targets[:, :, 0].to(torch.float64)
        field = torch.polar(torch.ones_like(phi0, dtype=torch.float64), phi0.to(torch.float64))
        ahat, res, err = [], [], []
        for k in range(4):
            u = propagate_fresnel(field, self._config.plane(k)).field[:, 0]          # [B,H,W] complex128
            a = u.abs()
            a = a / a.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
            r = t[:, k] - a
            unit = u / u.abs().clamp_min(1e-12)                                          # exp(i arg U_k), safe at |U|=0
            e = inverse_fresnel((r * unit)[:, None], self._config.plane(k))[:, 0]
            rms = e.abs().pow(2).mean(dim=(-2, -1), keepdim=True).sqrt()
            e = torch.where(rms > _RMS_EPS, e / torch.where(rms > _RMS_EPS, rms, torch.ones_like(rms)), torch.zeros_like(e))
            ahat.append(a); res.append(r); err += [e.real, e.imag]
        chans = [torch.sin(phi0[:, 0].double()), torch.cos(phi0[:, 0].double())] + [t[:, k] * 2 - 1 for k in range(4)] \
            + [x * 2 - 1 for x in ahat] + res + err
        return torch.stack(chans, dim=1).to(targets.dtype)

    def _corr_unet(self, x, hw):
        h, w = hw
        c = self.corr
        skips = []
        for i, enc in enumerate(c.encoders):
            x = enc(x if i == 0 else F.avg_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for red, dec in zip(c.reducers, c.decoders):
            skip = skips.pop()
            x = red(F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False))
            x = dec(torch.cat([x, skip], dim=1))
        return c.head(x)[..., :h, :w]

    def forward(self, targets):
        self._check(targets)
        phi0 = self.stage0(targets)
        feats = self.feedback_features(targets, phi0)
        delta = math.pi * torch.tanh(self._corr_unet(feats, targets.shape[-2:]))
        return _wrap(phi0 + delta)


def from_v2(v2_model, corr_channels=16):
    m = FeedbackStemUNet(base_channels=v2_model.base_channels, depth=v2_model.depth, corr_channels=corr_channels)
    missing, unexpected = m.load_state_dict(v2_model.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("corr.") for k in missing), (missing, unexpected)
    return m
