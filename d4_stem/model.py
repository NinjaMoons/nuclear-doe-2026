"""RESEARCH ONLY (R14 Stage B): STEM with synchronous exact-D4 (c64) TRAIN-time augmentation of the whole quartet.
forward in training (grad enabled): phi = g^-1( STEM( g(T) ) ) with g drawn per sample from a private generator.
Because F_z(g U) = g F_z(U) exactly (D4_PHYSICS_AUDIT) and the R3 objective is invariant under the same pixel
permutation applied to amplitudes and targets, the stock training loss equals the loss of training STEM on the
transformed quartet g(T) (tested). STEM features are recomputed from g(T) (no transformation of saved features).
Eval / no_grad: plain STEM (identity), so inference is the unchanged one-forward STEM."""
import torch
from back_projection_stem.stem import BackProjectionStemUNet
from d4_stem.d4 import ELEMENTS, apply, inverse

AUG_SEED_OFFSET = 4545


class D4AugStemUNet(BackProjectionStemUNet):
    def __init__(self, base_channels=16, depth=3, config=None, aug_seed=0):
        super().__init__(base_channels=base_channels, depth=depth, config=config)
        self._aug_gen = torch.Generator(device="cpu").manual_seed(AUG_SEED_OFFSET + int(aug_seed))
        self.last_words = None

    @property
    def hparams(self):
        hp = dict(super().hparams); hp.update({"train_aug": "d4_c64_sync_v0"}); return hp

    def forward(self, targets):
        if not (self.training and torch.is_grad_enabled()):
            return super().forward(targets)
        idx = torch.randint(0, len(ELEMENTS), (targets.shape[0],), generator=self._aug_gen).tolist()
        words = [ELEMENTS[i] for i in idx]
        self.last_words = words
        tg = torch.stack([apply(w, targets[i]) for i, w in enumerate(words)])
        phi = super().forward(tg)
        return torch.stack([apply(inverse(w), phi[i]) for i, w in enumerate(words)])


def build_d4_model(*, seed, base_channels=16, depth=3):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return D4AugStemUNet(base_channels=base_channels, depth=depth, aug_seed=seed)
