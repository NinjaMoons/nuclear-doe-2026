import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class STEQuantize8Bit(torch.autograd.Function):
    """Честное 8-битное квантование фазы [-pi, pi) со сквозным градиентом (STE)."""
    @staticmethod
    def forward(ctx, phase: torch.Tensor) -> torch.Tensor:
        # Перевод из [-pi, pi] в [0, 2*pi) -> 256 уровней -> обратно в [-pi, pi)
        twopi = 2.0 * math.pi
        p_norm = torch.remainder(phase + math.pi, twopi) / twopi
        p_q = torch.round(p_norm * 255.0) / 256.0
        return p_q * twopi - math.pi

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output

class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(8, ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(8, ch),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.net(x))

class GF_DDFN(nn.Module):
    """
    Gauge-Fixed Dual-Domain Fresnel Network (GF-DDFN).
    Ведущая роль нейросети + точный сопряженный подъем Single-FFT Fresnel + 2 режимные головы.
    """
    def __init__(
        self,
        N: int = 128,
        pitch: float = 10e-6,
        wavelength: float = 532e-9,
        z_list: tuple = (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3),
        width: int = 32,
        anchor_seed: int = 2026,
    ):
        super().__init__()
        self.N = N
        self.num_planes = len(z_list)

        # 1. Предвычисляем точные фазовые чирпы для Single-FFT Fresnel
        coords = (torch.arange(N, dtype=torch.float32) - N // 2) * pitch
        Y0, X0 = torch.meshgrid(coords, coords, indexing="ij")
        R0_sq = X0**2 + Y0**2  # [128, 128]

        in_chirps, out_chirps = [], []
        for z in z_list:
            dx_z = (wavelength * z) / (N * pitch)
            coords_z = (torch.arange(N, dtype=torch.float32) - N // 2) * dx_z
            Yz, Xz = torch.meshgrid(coords_z, coords_z, indexing="ij")
            Rz_sq = Xz**2 + Yz**2

            phi_in = (math.pi / (wavelength * z)) * R0_sq
            phi_out = (math.pi / (wavelength * z)) * Rz_sq
            in_chirps.append(torch.exp(-1j * phi_in))   # Сопряженный входной чирп D_in^*
            out_chirps.append(torch.exp(-1j * phi_out)) # Сопряженный выходной чирп D_out^*

        self.register_buffer("conj_in_chirp", torch.stack(in_chirps, dim=0).unsqueeze(0))   # [1, 4, 128, 128]
        self.register_buffer("conj_out_chirp", torch.stack(out_chirps, dim=0).unsqueeze(0)) # [1, 4, 128, 128]

        # 2. Фиксированный канонический ВЧ-фазовый якорь в целевых плоскостях (Gauge Fixing)
        gen = torch.Generator().manual_seed(anchor_seed)
        # Сумма квадратичного рассеивателя и полосно-ограниченного ВЧ-якоря
        quad_carrier = math.pi * (R0_sq / R0_sq.max()) * 16.0
        rand_hf = (torch.rand((1, 4, N, N), generator=gen) - 0.5) * (1.5 * math.pi)
        self.register_buffer("psi_anchor", quad_carrier.view(1, 1, N, N) + rand_hf)

        # 3. Стадия 1: Нейросеть предкомпенсации амплитуд и фазы в целевых плоскостях
        self.target_encoder = nn.Sequential(
            nn.Conv2d(4, width, 3, padding=1),
            nn.SiLU(inplace=True),
            ResBlock(width),
            ResBlock(width),
            ResBlock(width),
        )
        self.target_head = nn.Conv2d(width, 8, 1)  # 4 канала dA (cross-talk unmixing) + 4 канала dPsi
        nn.init.zeros_(self.target_head.weight)
        nn.init.zeros_(self.target_head.bias)

        # 4. Физический роутер режимов (оценивает межплоскостной конфликт входа T)
        # 6 попарных косинусов между 4 плоскостями + 4 отношения ВЧ/НЧ энергии = 10 признаков
        self.router = nn.Sequential(
            nn.Linear(10, 16),
            nn.SiLU(),
            nn.Linear(16, 2),  # 2 режима: эксперт A (когерентный) и эксперт B (высококонфликтный)
        )

        # 5. Стадия 2: Нейросеть синтеза фазы в плоскости апертуры DOE
        self.doe_trunk = nn.Sequential(
            nn.Conv2d(10, width, 3, padding=1, padding_mode="circular"),
            nn.SiLU(inplace=True),
            ResBlock(width),
            ResBlock(width),
            ResBlock(width),
        )
        # Две специализированные фазовые головы (предсказывают комплексный фазор dRe, dIm)
        self.head_expert_A = nn.Conv2d(width, 2, 3, padding=1, padding_mode="circular")
        self.head_expert_B = nn.Conv2d(width, 2, 3, padding=1, padding_mode="circular")

    def compute_regime_features(self, T: torch.Tensor) -> torch.Tensor:
        """Извлекает 10 инвариантных физических признаков межплоскостного смешения из входа T [B, 4, H, W]."""
        B = T.shape[0]
        flat = F.normalize(T.reshape(B, 4, -1), dim=-1)
        sim_mat = torch.bmm(flat, flat.transpose(1, 2))  # [B, 4, 4]
        idx_i, idx_j = torch.triu_indices(4, 4, offset=1, device=T.device)
        pair_sims = sim_mat[:, idx_i, idx_j]  # [B, 6]

        T_lp = F.avg_pool2d(T, kernel_size=5, stride=1, padding=2)
        hf_ratio = (T - T_lp).norm(dim=(-2, -1)) / T.norm(dim=(-2, -1)).clamp_min(1e-6)  # [B, 4]
        return torch.cat([pair_sims, hf_ratio], dim=-1)  # [B, 10]

    def adjoint_fresnel_lift(self, U_target: torch.Tensor) -> torch.Tensor:
        """Точный сопряженный подъем Single-FFT Fresnel из 4 плоскостей в плоскость DOE."""
        # U_target: [B, 4, 128, 128] complex64
        x = U_target * self.conj_out_chirp
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=(-2, -1))
        return x * self.conj_in_chirp  # [B, 4, 128, 128] complex64 в плоскости DOE

    def forward(self, T: torch.Tensor) -> torch.Tensor:
        # Шаг 1: Предкомпенсация амплитуд и фазовая коррекция в целевых плоскостях
        feat_t = self.target_encoder(T)
        delta_t = self.target_head(feat_t)
        dA, dPsi = torch.chunk(delta_t, 2, dim=1)

        A_comp = F.relu(T + 0.3 * torch.tanh(dA)) + 1e-4
        psi_target = self.psi_anchor + math.pi * torch.tanh(dPsi)
        U_target = torch.polar(A_comp, psi_target)  # [B, 4, 128, 128] complex64

        # Шаг 2: Точный аналитический перенос каждого поля в плоскость DOE
        U_doe_per_plane = self.adjoint_fresnel_lift(U_target)  # [B, 4, 128, 128] complex
        U_doe_sum = U_doe_per_plane.mean(dim=1, keepdim=True)  # [B, 1, 128, 128] complex
        U_doe_norm = U_doe_sum / U_doe_sum.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)

        # Вход для DOE-сети: Re/Im всех 4 поднятых плоскостей + Re/Im суммы = 10 каналов
        doe_in = torch.cat([
            U_doe_per_plane.real,
            U_doe_per_plane.imag,
            U_doe_norm.real,
            U_doe_norm.imag,
        ], dim=1)  # [B, 10, 128, 128]

        # Шаг 3: Нейросетевой синтез в плоскости DOE с режимным роутингом
        doe_feat = self.doe_trunk(doe_in)
        phasor_A = self.head_expert_A(doe_feat)
        phasor_B = self.head_expert_B(doe_feat)

        logits = self.router(self.compute_regime_features(T))  # [B, 2]
        if self.training:
            gates = F.gumbel_softmax(logits, tau=0.5, hard=False).view(-1, 2, 1, 1, 1)
            phasor_res = gates[:, 0] * phasor_A + gates[:, 1] * phasor_B
        else:
            # На конкурсном инференсе — детерминированный Top-1 выбор одной ветви
            idx = logits.argmax(dim=-1).view(-1, 1, 1, 1)
            phasor_res = torch.where(idx == 0, phasor_A, phasor_B)

        # Сложение комплексного фазора затравки и нейросетевой коррекции (без разрывов на +-pi)
        base_phasor = torch.cat([U_doe_norm.real, U_doe_norm.imag], dim=1)
        total_phasor = base_phasor + phasor_res
        phase = torch.atan2(total_phasor[:, 1:2], total_phasor[:, 0:1])

        return STEQuantize8Bit.apply(phase)  # [B, 1, 128, 128]

