"""RESEARCH ONLY (R18): complex back-projection mixing head on the 12-ch STEM (see PLAN_R18.md)."""
import torch, torch.nn.functional as F
from torch import nn
from back_projection_stem.stem import BackProjectionStemUNet, backprojection_features
G = 32


class BPMixStemUNet(BackProjectionStemUNet):
    def __init__(self, base_channels=16, depth=3, config=None):
        super().__init__(base_channels=base_channels, depth=depth, config=config)
        self.mix_head = nn.Conv2d(self.head.in_channels, 10, 1)
        with torch.no_grad():
            self.mix_head.weight.zero_(); self.mix_head.bias.zero_()

    @property
    def hparams(self):
        hp = dict(super().hparams); hp.update({"phase_head": "bpmix_v0", "mix_grid": G}); return hp

    def forward(self, targets):
        self._check(targets)
        h, w = targets.shape[-2:]
        feats = backprojection_features(targets, self._config)
        x = feats; m = 2 ** self.depth
        pad_h, pad_w = (-h) % m, (-w) % m
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        skips = []
        for i, enc in enumerate(self.encoders):
            x = enc(x if i == 0 else F.avg_pool2d(x, 2)); skips.append(x)
        x = skips.pop()
        for red, dec in zip(self.reducers, self.decoders):
            skip = skips.pop()
            x = red(F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False))
            x = dec(torch.cat([x, skip], dim=1))
        x = x[..., :h, :w]
        phi = self.head(x)                                                     # [B,1,H,W] unwrapped STEM phase
        a = F.interpolate(self.mix_head(F.adaptive_avg_pool2d(x, (G, G))), size=(h, w), mode="bilinear", align_corners=False)
        c0 = torch.complex(1 + a[:, 0:1], a[:, 1:2])
        s = c0 * torch.exp(1j * phi)
        for k in range(4):
            bk = torch.complex(feats[:, 4 + 2 * k:5 + 2 * k], feats[:, 5 + 2 * k:6 + 2 * k])
            s = s + torch.complex(a[:, 2 + 2 * k:3 + 2 * k], a[:, 3 + 2 * k:4 + 2 * k]) * bk
        return torch.angle(s)


def from_v2(v2_model):
    m = BPMixStemUNet(base_channels=v2_model.base_channels, depth=v2_model.depth)
    missing, unexpected = m.load_state_dict(v2_model.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("mix_head.") for k in missing), (missing, unexpected)
    return m
