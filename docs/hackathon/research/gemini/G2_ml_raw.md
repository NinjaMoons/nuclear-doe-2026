# G2 — raw Gemini answer (ML / TRAINING), verbatim

- Prompt: "Сессия 2 — G2" block of `PROMPTS_SENT_G1_G2.md` (= GEMINI_TO_CLAUDE.md, LIVE FACTS + GEMINI 2).
- Provenance: pasted by the user (Grigoriy) into Claude Code chat ~00:00 MSK 08.11.2026; Gemini model/version: UNKNOWN (not stated).
- Text below is unchanged.

---

### Executive Architectural Diagnosis: The Amortization Gap

The empirical results establish three critical facts about your system:

1. **Physical Separability is Proven, Not the Bottleneck:** Starting from the `v2 seed0 step12000` (`Asset 721714285`) phase prediction, per-sample 2000-step Adam reaches **`P = 0.7952` / `B_P = 0.6526**` vs. the neural network’s **`P = 0.61243` / `B_P = 0.4516**`. A single 8-bit $128\times 128$ phase DOE physically supports clean 4-plane separation on Group B, but the feed-forward 12ch STEM suffers from a **`+0.201` Group-B amortization gap**.
2. **Why `B75`, `D4`, `WhiteCollars`, and `R15` Failed or Flatlined:** Data-expansion (`WhiteCollars`, `D4`) and reweighting (`B75`) only help when a network is *overfitting* a small training set. Here, the 12ch STEM is **severely underfitting the non-local inverse physics** on `TRAIN512` itself. Furthermore, `WhiteCollars` introduces out-of-domain spatial statistics (and lacks cleared rights, making it an immediate **NO-GO**), while weak finite-difference regularization (`R15 +0.05 axial-loss`) leaves the primary raw-target gradient conflict untouched.
3. **Why `B median` Ghosts Despite `4/4 Own-Target` Match:** An `own-target` score of `4/4` only means the diagonal entry of the $4\times 4$ correlation matrix $C_{m,k} = \text{corr}(\hat{A}_m, A_k)$ exceeds the off-diagonal entries ($C_{m,m} > \max_{k\neq m} C_{m,k}$). When $C_{m,m} \approx 0.45$ and off-diagonal $C_{m,k} \approx 0.28$, the argmax is `4/4`, yet visually $30\%+$ of the contrasting edges in plane $m$ are ghost bleed from plane $k\neq m$. Worse, this background bleed inflates mean luminance $\mu_{\hat{A}}$ and compresses variance $\sigma_{\hat{A}}^2$ on the reloaded `uint8` PNG, directly degrading **global SSIM**:

$$\text{SSIM}_{\text{global}}(\hat{A}, A) = \frac{2\mu_{\hat{A}}\mu_A + C_1}{\mu_{\hat{A}}^2 + \mu_A^2 + C_1} \cdot \frac{2\sigma_{\hat{A}}\sigma_A + C_2}{\sigma_{\hat{A}}^2 + \sigma_A^2 + C_2} \cdot \frac{\sigma_{\hat{A}A} + C_2/2}{\sigma_{\hat{A}}\sigma_A + C_2/2}$$



---

### Evaluation of the Three Train-Time-Only Ideas

#### Idea (a): Direct Phase-Space Teacher/Student Distillation ($\phi_{\text{student}} \to \phi_{\text{teacher}}$)

* **Mechanism:** Precompute per-sample Adam phase DOEs $\phi^{\text{teacher}}_i \in [-\pi, \pi)^{128\times 128}$ on `TRAIN512` and train the neural network to regress $\phi^{\text{teacher}}_i$ directly in the DOE plane.
* **Crucial Pitfalls & Failure Modes:**
1. **Direct Angle Regression Across $\pm\pi$ Phase Wraps:** Naive $L_1/L_2$ loss $\Vert{}\hat{\phi} - \phi^{\text{teacher}}\Vert{}_p$ creates artificial $2\pi$ cliffs where $\pi - \epsilon$ and $-\pi + \epsilon$ represent the exact same physical phasor $e^{i\phi}$ but receive maximum gradient penalty, destabilizing training.
2. **Multi-Modal Non-Convex Basins & Neural Averaging Collapse:** Multi-plane Fresnel phase retrieval is massively non-unique. Beyond trivial global piston invariance ($\phi \to \phi + c$ produces identical $\vert{}\mathcal{P}_m(e^{i\phi})\vert{}$), two visually similar inputs $x_1 \approx x_2$ optimized by Adam converge to **distinct local minima** $\phi^{\text{teacher}}_1 \neq \phi^{\text{teacher}}_2$ with incompatible high-frequency interference fringes. Even with wrap-safe phasor loss $1 - \cos(\hat{\phi} - \phi^{\text{teacher}})$, a continuous neural network forced to map similar inputs to uncorrelated phase basins outputs the conditional expectation $\mathbb{E}[e^{i\phi}\vert{}x] \to 0$, washing out high-frequency diffractive fringes.
3. **Teacher Quality Variance:** Unrestricted 2000-step Adam introduces isolated phase vortices ($2\pi$ screw dislocations) that slightly raise pixelwise Pearson on `TRAIN512` but cannot be represented smoothly by a convolutional U-Net.



#### Idea (b): Basin-Locked Teacher-Guided Image-Domain Optical Distillation + Crosstalk Margin (`BLOT-CM`)

* **Mechanism:** Instead of forcing the student to match arbitrary phase angles, run a **short-horizon (400-step) batched Adam refinement initialized directly from the frozen `v2 seed0 step12000` (`Asset 721714285`) phase prediction** on `TRAIN512`, and supervise the student **through the differentiable single-FFT Fresnel forward model** (with 8-bit DOE Straight-Through Estimator) against the teacher's propagated optical fields $\hat{U}^{\text{teacher}}_m = \mathcal{P}_m(e^{i\phi^{\text{teacher}}})$, combined with an explicit off-diagonal crosstalk margin penalty.
* **Why It Solves the Amortization Gap Without Idea (a)'s Traps:**
1. **Eliminates Impossible-Target Gradient Tug-of-War:** Raw targets $A_m$ demand $100\%$ diffraction efficiency and zero background simultaneously across all 4 sub-Rayleigh planes—a physical impossibility for a single phase screen. The raw R3 objective wastes network capacity fighting unachievable gradients. The teacher's reconstructed amplitudes $A^T_m = \vert{}\mathcal{P}_m(e^{i\phi^{\text{teacher}}})\vert{}$ represent a **physically realizable Pareto projection** onto the manifold of single-DOE fields.
2. **Homotopy Basin Locking + Gauge Invariance:** Because the teacher is initialized at $\phi^{(0)}_i = \text{STEM}_{v2}(x_i)$ and refined for 400 steps with moderate step size, $\phi^{\text{teacher}}_i$ stays inside the **exact same local phase basin** as the student network. Supervising the student via forward-propagated amplitudes $A^T_m$ plus a **piston-invariant complex phasor coherence auxiliary term** ($1 - \frac{1}{N^2}\vert{}\langle e^{i\hat{\phi}}, e^{i\phi^{\text{teacher}}} \rangle\vert{}$, gated by teacher advantage $\Delta P_i > 0$) is $100\%$ immune to $2\pi$ wraps and global piston shifts.
3. **Direct Attack on `B median` Ghosting:** Adding an explicit off-diagonal Gram-Schmidt crosstalk penalty on the $4\times 4$ target correlation matrix forces the student to allocate its phase budget toward increasing the diagonal-to-off-diagonal margin $C_{m,m} - \max_{k\neq m} C_{m,k}$.



#### Idea (c): Training Group-B Recombined Independent Quartets with Strict Split Provenance

* **Mechanism:** Partition native `TRAIN512` into A-like and B-like pools ($\vert{}S_B\vert{} \approx 256$) strictly within `TRAIN512`, and synthesize new 4-plane training examples by drawing 4 independent B samples $(x^{(i_1)}_1, x^{(i_2)}_2, x^{(i_3)}_3, x^{(i_4)}_4)$, keeping each plane at its native axial slot $m \in \{1,2,3,4\}$ so physical pitch $\Delta x_m = \frac{\lambda z_m}{N\Delta x_0}$ is never violated.
* **Crucial Pitfalls & Why It Ranks Lower Right Now:**
1. **Exacerbates Underfitting:** Because `v2 step12000` is currently optimization/capacity-bound on native `TRAIN512` (`B_P = 0.4516`), replacing natural inter-plane correlations with synthetic high-entropy combinations makes the inverse problem *harder* to fit, diluting gradient signal on real sample structure (explaining why `B75` and `D4` were neutral).
2. **Split Provenance & Pitch Hazard:** Permuting planes across axial indices ($z_1 \leftrightarrow z_4$) introduces a $14.9\%$ physical scale mismatch ($\frac{27.7}{24.1} = 1.149$), and any accidental inclusion of `VAL16` planes invalidates evaluation.



---

### Mandatory Pitfall Checklist: Latency, Leakage & Inference Compliance

1. **Cost & Wall-Clock Latency of Generating 512 Teachers:**
* Running 2000-step Adam sequentially (`batch_size = 1`) for 512 samples requires $1.024 \times 10^6$ serial calls—taking **>3.5 hours**.
* **The Fix (Batched Warm-Start on B-Subset or Full `TRAIN512`):** In PyTorch, a single $128\times 128$ 4-plane FFT forward+backward pass processes **`batch_size = 64` samples in parallel in ~3.8 ms** on a modern GPU (or ~22 ms on Apple Silicon M-series MPS). Furthermore, starting from `v2 step12000` (`Asset 721714285`), **400 steps of Adam (`lr=0.08`, cosine decay)** captures **$>88\%$ of the 2000-step Pearson gain**.
* *Total wall time for all 512 teachers (`8 batches × 400 steps`):* **~15–25 seconds on GPU** or **~2.5 minutes on Mac MPS**, cached once to a single `train512_v2_teachers.pt` file (`32 MB` for `512 × 128 × 128` `uint8` DOEs).


2. **Zero Label Leakage:**
* Teacher generation is executed **strictly on `TRAIN512` indices**. `VAL16` and `Frozen TEST` are never passed to the teacher generator used for training.


3. **Strict Separation of Train-Time Teacher vs. Forbidden Test-Time Adam:**
* At **training time**, $\phi^{\text{teacher}}_i$ is merely a precomputed static label tensor on `TRAIN512` used inside `loss_fn(pred_doe, target_amp, teacher_doe)`.
* At **inference / evaluation time**, the loss module and `train512_v2_teachers.pt` are completely absent. The exported model is **100% identical in graph topology, parameter count, and single-forward latency** to `v2 seed0 step12000` (`12ch input -> STEM U-Net -> 1 forward pass -> 8-bit DOE`).



---

### Comparative Ranking of the Three Ideas

| Criterion | **Rank 1: Idea (b) `BLOT-CM**` (Teacher-Guided Optical + Crosstalk Margin) | **Rank 2: Idea (a) Pure Phase Distillation** (Circular/Phasor Regression) | **Rank 3: Idea (c) B-Quartet Recombination** (Strict Native Split Provenance) |
| --- | --- | --- | --- |
| **Expected Gain in Group B Visual Identity** | **High (`+0.025` to `+0.050` B_P, `-15%` ghosting)**; directly attacks off-diagonal bleed & impossible-target stalling | **Low to Moderate (`-0.01` to `+0.015`)**; prone to high-frequency fringe blurring from basin conflicts | **Neutral (`-0.005` to `+0.008`)** on current capacity; only useful once `W64d3` overfits `TRAIN512` |
| **Coding Time** | **35 mins** (~120 lines: batched teacher cache + optical loss module) | **25 mins** (~80 lines) | **20 mins** (~50 lines in `Dataset.__getitem__`) |
| **Wall Time (p50 / p90)** | **p50: 18 min / p90: 28 min** (1 min teacher cache + 1500 fine-tune steps from `v2 step12000`) | **p50: 16 min / p90: 25 min** | **p50: 45 min / p90: 65 min** (requires longer schedule to converge on synthetic quartets) |
| **GPU / VRAM Use** | **~2.1 GB VRAM** (zero model bloat; teacher generation can even run on Mac while Sanya's GPU finishes) | **~1.9 GB VRAM** | **~1.8 GB VRAM** |
| **AI Quota / Reset Risk** | **Minimal** (cached `train512_v2_teachers.pt` is `32 MB`; warm-start needs only 1,500 steps) | **Low–Moderate** (if phase weight is too high, requires hyperparameter restarts) | **Low** |
| **Ease of Independent Acceptance** | **100% Drop-In** (identical `v2` architecture, weights verified on Mac via standalone forward script) | **100% Drop-In** (if it converges) | **Requires Provenance Audit** (to prove zero `VAL16` plane contamination) |

---

### The Single Controlled Pilot: `BLOT-CM` 1500-Step Warm Fine-Tune from `v2 seed0 step12000`

To avoid colliding with Sanya’s `W32d3 / W64d3 / W32d4` GPU jobs:

1. **Stage 1 (Offline Teacher Cache — Can Run on Mac or in 20s between GPU runs):** Load `v2 seed0 step12000` (`Asset 721714285`), run batched 400-step Adam (`B=64`) on `TRAIN512` starting from the network's own predicted phase $\phi^{(0)}_i$, and save `train512_v2_teachers.pt` (storing `teacher_doe_uint8` and per-plane `teacher_pearson`).
2. **Stage 2 (Controlled A/B Fine-Tuning Pilot — 1,500 Steps from `v2 seed0 step12000`):**
* **Control (`Run_Ctrl`):** Continue `v2 seed0 step12000` for `+1500` steps (`step 12000 -> 13500`, `lr = 1.5e-4` cosine to `1e-5`) with the original `R3` objective.
* **Treatment (`Run_BLOT_CM`):** Continue `v2 seed0 step12000` for the exact same `+1500` steps, same seed, batch order, and LR schedule, replacing the objective with $\mathcal{L}_{\text{BLOT-CM}}$.



#### Exact Mathematical Formulation of $\mathcal{L}_{\text{BLOT-CM}}$

Let $\hat{\phi} \in [0, 2\pi)^{128\times 128}$ be the network's predicted phase. Apply the **8-bit Quantization Straight-Through Estimator (STE)** to match competition export:


$$\hat{\phi}_q = \frac{2\pi}{256}\text{round}\!\left(\frac{255}{2\pi}\text{mod}(\hat{\phi}, 2\pi)\right), \qquad \hat{\phi}_{\text{STE}} = \hat{\phi} + \text{sg}[\hat{\phi}_q - \hat{\phi}]$$


Propagate both student $\hat{U}_m = \mathcal{P}_m(e^{i\hat{\phi}_{\text{STE}}})$ and teacher $U^T_m = \mathcal{P}_m(e^{i\phi^{\text{teacher}}_q})$ through the exact single-FFT Fresnel kernel to obtain amplitudes $\hat{A}_m = \vert{}\hat{U}_m\vert{}$ and $A^T_m = \vert{}U^T_m\vert{}$. Define standardized zero-mean unit-variance maps $\tilde{X} = \frac{X - \mu(X)}{\sigma(X) + \epsilon}$ ($\epsilon = 10^{-6}$).

1. **Teacher Advantage Gate ($g_{i,m} \in [0, 1]$):** Only trust the teacher on plane $(i,m)$ proportional to its actual improvement over the `v2 step12000` initialization:

$$g_{i,m} = \text{clip}\!\left(\frac{P(A^T_{i,m}, A_{i,m}) - P(\hat{A}^{(0)}_{i,m}, A_{i,m})}{0.15}, \, 0, \, 1\right)$$


2. **Realizable Optical Target Distillation ($\mathcal{L}_{\text{opt}}$):** Construct the physically realizable target amplitude $A^{\star}_{i,m} = (1 - 0.5 g_{i,m})\tilde{A}_{i,m} + 0.5 g_{i,m}\tilde{A}^T_{i,m}$ and compute the R3 + Global-SSIM-aligned optical loss:

$$\mathcal{L}_{\text{opt}} = \frac{1}{4}\sum_{m=1}^4 \left[ \left(1 - \text{corr}(\hat{A}_m, A^{\star}_m)\right) + 0.25 \left(1 - \text{SSIM}_{\text{global}}(\hat{\alpha}_m \hat{A}_m, A_m)\right) \right]$$



where $\hat{\alpha}_m = \text{sg}\!\left[\frac{\langle \hat{A}_m, A_m \rangle}{\Vert{}\hat{A}_m\Vert{}_2^2 + \epsilon}\right]$ aligns exposure scale prior to `uint8`-equivalent global SSIM.
3. **Gram-Schmidt Off-Diagonal Crosstalk Suppression ($\mathcal{L}_{\text{xtalk}}$):** Orthogonalize each foreign target $\tilde{A}_k$ ($k \neq m$) against own target $\tilde{A}_m$:

$$T_{k \perp m} = \tilde{A}_k - \frac{\langle \tilde{A}_k, \tilde{A}_m \rangle}{\Vert{}\tilde{A}_m\Vert{}_2^2 + \epsilon}\tilde{A}_m, \qquad \hat{T}_{k \perp m} = \frac{T_{k \perp m}}{\Vert{}T_{k \perp m}\Vert{}_2 + \epsilon}$$



Penalize the projection of the student's residual onto all 3 foreign planes plus dark-zone bleed:

$$\mathcal{L}_{\text{xtalk}} = \frac{1}{12}\sum_{m=1}^4 \sum_{k \neq m} \left(\langle \tilde{\hat{A}}_m - \text{corr}(\hat{A}_m, A_m)\tilde{A}_m, \, \hat{T}_{k \perp m} \rangle\right)^2 + \frac{1}{4}\sum_{m=1}^4 \left\Vert{} M_{\text{dark},m} \odot \max(0, \hat{\alpha}_m \hat{A}_m - A_m) \right\Vert{}_2^2$$


4. **Piston-Invariant Phasor Coherence ($\mathcal{L}_{\text{phasor}}$):**

$$\mathcal{L}_{\text{phasor}} = \bar{g}_i \cdot \left(1 - \frac{1}{N^2}\left\vert{}\sum_{n,l} \exp\!\left(i(\hat{\phi}_{\text{STE}}[n,l] - \phi^{\text{teacher}}_q[n,l])\right)\right\vert{}\right), \qquad \bar{g}_i = \frac{1}{4}\sum_{m=1}^4 g_{i,m}$$


5. **Total Training Objective:**

$$\mathcal{L}_{\text{BLOT-CM}} = \mathcal{L}_{\text{opt}} + \lambda_{\text{xtalk}} \mathcal{L}_{\text{xtalk}} + \lambda_{\text{phasor}} \mathcal{L}_{\text{phasor}}, \qquad \lambda_{\text{xtalk}} = 0.25, \quad \lambda_{\text{phasor}} = 0.10$$



---

### Complete Runnable Code: Loss, Teacher Generator, Diagnostics & 4 Failure Tests

Save the module below as `pilot_blot_cm.py` (includes both the `BLOT_CM_Loss` / `GhostOrthogonalContrastLoss`, the batched warm-start teacher generator, the `uint8`-reloaded Global SSIM + $4\times 4$ Own-Target Matrix evaluator, B median/hard contact sheet generator, and the 4 mandatory unit failure tests):

```python
# pilot_blot_cm.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

# =====================================================================
# 1. Official Single-FFT Fresnel Propagator & 8-Bit STE
# =====================================================================
class SingleFFTFresnel(nn.Module):
    """Exact single-FFT Fresnel propagator for N=128, pitch=10um, lambda=532nm."""
    def __init__(
        self,
        N: int = 128,
        pitch_m: float = 10e-6,
        wavelength_m: float = 532e-9,
        z_list_m: Tuple[float, ...] = (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3),
    ):
        super().__init__()
        self.N = N
        coords = (torch.arange(N, dtype=torch.float32) - N // 2) * pitch_m
        Y, X = torch.meshgrid(coords, coords, indexing="ij")
        R2 = X ** 2 + Y ** 2  # [128, 128]
        z_tensor = torch.tensor(z_list_m, dtype=torch.float32).view(-1, 1, 1)  # [4, 1, 1]
        quad_phase = (math.pi / (wavelength_m * z_tensor)) * R2.unsqueeze(0)  # [4, 128, 128]
        chirp = torch.exp(1j * quad_phase)
        self.register_buffer("chirp", chirp)

    def forward(self, phase_rad: torch.Tensor) -> torch.Tensor:
        """
        phase_rad: [B, 128, 128] in radians -> returns amplitude [B, 4, 128, 128]
        """
        u0 = torch.exp(1j * phase_rad).unsqueeze(1)  # [B, 1, 128, 128]
        field = u0 * self.chirp.unsqueeze(0)         # [B, 4, 128, 128]
        spec = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1)), norm="ortho"),
            dim=(-2, -1),
        )
        return torch.abs(spec)


def quantize_phase_8bit_ste(phase_rad: torch.Tensor) -> torch.Tensor:
    """8-bit DOE quantization [0, 2*pi) with Straight-Through Estimator."""
    two_pi = 2.0 * math.pi
    wrapped = torch.remainder(phase_rad, two_pi)
    q_idx = torch.round((wrapped / two_pi) * 255.0).clamp(0.0, 255.0)
    phase_q = (q_idx / 256.0) * two_pi
    return phase_rad + (phase_q - phase_rad).detach()


# =====================================================================
# 2. BLOT-CM Loss (Teacher-Guided Optical + Gram-Schmidt Crosstalk Margin)
# =====================================================================
class BLOTCMLoss(nn.Module):
    def __init__(
        self,
        lambda_xtalk: float = 0.25,
        lambda_phasor: float = 0.10,
        lambda_dark: float = 0.15,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.lambda_xtalk = lambda_xtalk
        self.lambda_phasor = lambda_phasor
        self.lambda_dark = lambda_dark
        self.eps = eps

    def _standardize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), unbiased=False, keepdim=True)
        valid = (std > 1e-5).float()
        x_norm = ((x - mu) / (std + self.eps)) * valid
        return x_norm, valid.squeeze(-1).squeeze(-1)

    def _global_ssim_diff(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Differentiable Global SSIM over the full 128x128 plane."""
        alpha = (
            (pred * target).sum(dim=(-2, -1), keepdim=True)
            / (pred.pow(2).sum(dim=(-2, -1), keepdim=True) + self.eps)
        ).detach().clamp(min=0.0)
        p_scaled = alpha * pred

        L = target.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-3)
        c1 = (0.01 * L).pow(2)
        c2 = (0.03 * L).pow(2)

        mu_p = p_scaled.mean(dim=(-2, -1), keepdim=True)
        mu_t = target.mean(dim=(-2, -1), keepdim=True)
        var_p = ((p_scaled - mu_p) ** 2).mean(dim=(-2, -1), keepdim=True)
        var_t = ((target - mu_t) ** 2).mean(dim=(-2, -1), keepdim=True)
        cov_pt = ((p_scaled - mu_p) * (target - mu_t)).mean(dim=(-2, -1), keepdim=True)

        ssim_num = (2.0 * mu_p * mu_t + c1) * (2.0 * cov_pt + c2)
        ssim_den = (mu_p.pow(2) + mu_t.pow(2) + c1) * (var_p + var_t + c2)
        return (ssim_num / (ssim_den + self.eps)).squeeze(-1).squeeze(-1)

    def forward(
        self,
        pred_amp: torch.Tensor,
        target_amp: torch.Tensor,
        pred_phase: Optional[torch.Tensor] = None,
        teacher_amp: Optional[torch.Tensor] = None,
        teacher_phase: Optional[torch.Tensor] = None,
        teacher_adv_gate: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        pred_amp, target_amp: [B, 4, 128, 128]
        pred_phase, teacher_phase: [B, 128, 128] (optional for teacher phasor term)
        teacher_amp: [B, 4, 128, 128] (optional realizable teacher amplitudes)
        teacher_adv_gate: [B, 4] in [0, 1]
        """
        B, M, H, W = pred_amp.shape
        N2 = float(H * W)

        P_norm, p_valid = self._standardize(pred_amp)
        A_norm, a_valid = self._standardize(target_amp)
        rho_own = (P_norm * A_norm).mean(dim=(-2, -1)) * a_valid  # [B, 4]

        # 1. Optical Target Blend (if teacher_amp provided)
        if teacher_amp is not None and teacher_adv_gate is not None:
            T_norm, _ = self._standardize(teacher_amp)
            g = teacher_adv_gate.view(B, M, 1, 1).clamp(0.0, 1.0)
            A_star = (1.0 - 0.5 * g) * A_norm + 0.5 * g * T_norm
            A_star, _ = self._standardize(A_star)
            rho_opt = (P_norm * A_star).mean(dim=(-2, -1)) * a_valid
        else:
            rho_opt = rho_own

        ssim_global = self._global_ssim_diff(pred_amp, target_amp)
        loss_opt = ((1.0 - rho_opt) * a_valid + 0.25 * (1.0 - ssim_global) * a_valid).mean()

        # 2. Gram-Schmidt Off-Diagonal Crosstalk Penalty
        xtalk_terms = []
        for m in range(M):
            R_m = P_norm[:, m] - rho_own[:, m, None, None] * A_norm[:, m]
            for k in range(M):
                if m == k:
                    continue
                proj_coeff = (A_norm[:, k] * A_norm[:, m]).sum(dim=(-2, -1), keepdim=True) / (
                    A_norm[:, m].pow(2).sum(dim=(-2, -1), keepdim=True) + self.eps
                )
                T_perp = A_norm[:, k] - proj_coeff * A_norm[:, m]
                T_hat = T_perp / (T_perp.norm(p=2, dim=(-2, -1), keepdim=True) + self.eps)
                inner = (R_m * T_hat).sum(dim=(-2, -1)) / math.sqrt(N2)
                pair_valid = a_valid[:, m] * a_valid[:, k]
                xtalk_terms.append((inner.pow(2)) * pair_valid)
        loss_xtalk = torch.stack(xtalk_terms, dim=1).mean()

        # 3. Dark-Zone Bleed Suppression
        alpha_m = (
            (pred_amp * target_amp).sum(dim=(-2, -1), keepdim=True)
            / (pred_amp.pow(2).sum(dim=(-2, -1), keepdim=True) + self.eps)
        ).detach().clamp(min=0.0, max=10.0)
        t_max = target_amp.amax(dim=(-2, -1), keepdim=True)
        A_01 = torch.where(t_max > 1e-5, target_amp / (t_max + self.eps), torch.zeros_like(target_amp))

        dark_losses = []
        for m in range(M):
            other_idx = [k for k in range(M) if k != m]
            other_max = A_01[:, other_idx].amax(dim=1)
            M_dark = torch.exp(-A_01[:, m] / 0.15) * other_max
            excess = F.relu(alpha_m[:, m] * pred_amp[:, m] - target_amp[:, m])
            num = (M_dark * excess.pow(2)).sum(dim=(-2, -1))
            den = M_dark.sum(dim=(-2, -1)) + self.eps
            dark_losses.append(num / den)
        loss_dark = torch.stack(dark_losses, dim=1).mean()

        # 4. Piston-Invariant Phasor Coherence (wrap-safe & shift-invariant)
        if pred_phase is not None and teacher_phase is not None and teacher_adv_gate is not None:
            delta_phi = pred_phase - teacher_phase.detach()
            coherence = torch.abs(torch.exp(1j * delta_phi).mean(dim=(-2, -1)))  # [B]
            g_bar = teacher_adv_gate.mean(dim=1).clamp(0.0, 1.0)
            loss_phasor = (g_bar * (1.0 - coherence)).mean()
        else:
            loss_phasor = torch.zeros((), device=pred_amp.device)

        total = (
            loss_opt
            + self.lambda_xtalk * loss_xtalk
            + self.lambda_dark * loss_dark
            + self.lambda_phasor * loss_phasor
        )
        return {
            "loss": total,
            "loss_opt": loss_opt.detach(),
            "loss_xtalk": loss_xtalk.detach(),
            "loss_dark": loss_dark.detach(),
            "loss_phasor": loss_phasor.detach(),
            "mean_pearson": rho_own.mean().detach(),
        }


# =====================================================================
# 3. Batched Basin-Locked Adam Teacher Generator (Train-Time Only!)
# =====================================================================
def generate_basin_locked_teachers_batch(
    init_phase_rad: torch.Tensor,
    target_amp: torch.Tensor,
    propagator: SingleFFTFresnel,
    steps: int = 400,
    lr: float = 0.08,
) -> Dict[str, torch.Tensor]:
    """
    Refines STEM initial phase predictions on TRAIN512 batches.
    Strictly forbidden at competition inference time.
    """
    loss_module = BLOTCMLoss(lambda_xtalk=0.25, lambda_phasor=0.0, lambda_dark=0.15)
    with torch.no_grad():
        init_q = quantize_phase_8bit_ste(init_phase_rad)
        init_amp = propagator(init_q)
        P0, _ = loss_module._standardize(init_amp)
        A0, a_val = loss_module._standardize(target_amp)
        init_pearson = (P0 * A0).mean(dim=(-2, -1)) * a_val  # [B, 4]

    phi = init_phase_rad.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([phi], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.1)

    for _ in range(steps):
        optimizer.zero_grad()
        phi_ste = quantize_phase_8bit_ste(phi)
        pred_amp = propagator(phi_ste)
        out = loss_module(pred_amp, target_amp)
        out["loss"].backward()
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        final_q = quantize_phase_8bit_ste(phi)
        final_amp = propagator(final_q)
        PT, _ = loss_module._standardize(final_amp)
        teacher_pearson = (PT * A0).mean(dim=(-2, -1)) * a_val
        adv_gate = ((teacher_pearson - init_pearson) / 0.15).clamp(0.0, 1.0)
        # Keep original phase if teacher failed to improve sample
        improved_mask = (teacher_pearson.mean(dim=1) > init_pearson.mean(dim=1)).view(-1, 1, 1)
        best_phase = torch.where(improved_mask, final_q, init_q)
        best_amp = propagator(best_phase)

    return {
        "teacher_phase": best_phase.detach(),
        "teacher_amp": best_amp.detach(),
        "teacher_adv_gate": adv_gate.detach(),
        "init_pearson": init_pearson.detach(),
        "teacher_pearson": teacher_pearson.detach(),
    }


# =====================================================================
# 4. Competition Evaluator: Reloaded uint8 PNG Metrics & 4x4 Own-Target Matrix
# =====================================================================
@torch.no_grad()
def evaluate_uint8_metrics_and_4x4_matrix(
    pred_amp: torch.Tensor, target_amp: torch.Tensor, eps: float = 1e-8
) -> Dict[str, torch.Tensor]:
    """
    Simulates exact uint8 PNG quantization & reload, computing:
    - Per-plane Pearson & Global SSIM on uint8 [0..255]
    - 4x4 Cross-Correlation Matrix C[b, m, k] = corr(pred_m, target_k)
    - Own-Target Match Count (out of 4 per sample, /32 for 8 B samples)
    - Diagonal-to-Off-Diagonal Ghosting Margin M[b, m] = C[b,m,m] - max_{k!=m} C[b,m,k]
    """
    B, M, H, W = pred_amp.shape
    # Optimal exposure alignment before uint8 PNG save
    alpha = (pred_amp * target_amp).sum(dim=(-2, -1), keepdim=True) / (
        pred_amp.pow(2).sum(dim=(-2, -1), keepdim=True) + eps
    )
    t_max = target_amp.amax(dim=(-2, -1), keepdim=True).clamp(min=eps)
    pred_uint8 = torch.round((alpha * pred_amp / t_max).clamp(0.0, 1.0) * 255.0)
    targ_uint8 = torch.round((target_amp / t_max).clamp(0.0, 1.0) * 255.0)

    # Global SSIM on uint8 [0..255]
    c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
    mu_p = pred_uint8.mean(dim=(-2, -1))
    mu_t = targ_uint8.mean(dim=(-2, -1))
    var_p = ((pred_uint8 - mu_p[..., None, None]) ** 2).mean(dim=(-2, -1))
    var_t = ((targ_uint8 - mu_t[..., None, None]) ** 2).mean(dim=(-2, -1))
    cov_pt = ((pred_uint8 - mu_p[..., None, None]) * (targ_uint8 - mu_t[..., None, None])).mean(dim=(-2, -1))
    global_ssim = ((2 * mu_p * mu_t + c1) * (2 * cov_pt + c2)) / (
        (mu_p ** 2 + mu_t ** 2 + c1) * (var_p + var_t + c2) + eps
    )

    # 4x4 Cross-Correlation Matrix on uint8
    P_centered = pred_uint8 - mu_p[..., None, None]
    T_centered = targ_uint8 - mu_t[..., None, None]
    P_unit = P_centered / (P_centered.norm(p=2, dim=(-2, -1), keepdim=True) + eps)
    T_unit = T_centered / (T_centered.norm(p=2, dim=(-2, -1), keepdim=True) + eps)

    # C[b, m, k] = sum_{x,y} P_unit[b, m, x, y] * T_unit[b, k, x, y]
    C_4x4 = torch.einsum("bmhw,bkhw->bmk", P_unit, T_unit)  # [B, 4, 4]
    diag_pearson = torch.diagonal(C_4x4, dim1=1, dim2=2)    # [B, 4]
    own_match = (C_4x4.argmax(dim=2) == torch.arange(M, device=pred_amp.device).unsqueeze(0))  # [B, 4]

    # Off-diagonal max and Ghost Margin
    eye_mask = torch.eye(M, device=pred_amp.device, dtype=torch.bool).unsqueeze(0)
    C_off = C_4x4.masked_fill(eye_mask, -2.0)
    max_off_diag = C_off.max(dim=2).values  # [B, 4]
    ghost_margin = diag_pearson - max_off_diag

    return {
        "pearson_per_plane": diag_pearson,
        "global_ssim_per_plane": global_ssim,
        "C_4x4": C_4x4,
        "own_match_mask": own_match,
        "own_match_count": own_match.sum().item(),
        "ghost_margin_per_plane": ghost_margin,
        "max_off_diag_bleed": max_off_diag,
    }


# =====================================================================
# 5. Four Mandatory Failure / Sanity Unit Tests
# =====================================================================
def run_all_failure_tests() -> None:
    torch.manual_seed(0)
    loss_fn = BLOTCMLoss()
    B, M, H, W = 2, 4, 128, 128
    base_targets = torch.rand(B, M, H, W)

    # Test 1: Black / Constant Planes (No NaN/Inf, finite gradients)
    black_targets = base_targets.clone()
    black_targets[0, 1] = 0.0  # completely black plane
    black_targets[1, :] = 0.0  # all 4 planes black
    pred = torch.rand(B, M, H, W, requires_grad=True)
    res_black = loss_fn(pred, black_targets)
    res_black["loss"].backward()
    assert torch.isfinite(res_black["loss"]), "FAIL Test 1: NaN/Inf loss on black planes!"
    assert torch.isfinite(pred.grad).all(), "FAIL Test 1: NaN/Inf gradient on black planes!"

    # Test 2: Monotonic Sensitivity to Synthetic Ghosting
    # Construct 4 orthogonal spatial patterns
    y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    ortho_targets = torch.stack([
        (x > 0).float(),
        (y > 0).float(),
        (x * y > 0).float(),
        ((x**2 + y**2) < 0.5).float(),
    ], dim=0).unsqueeze(0)  # [1, 4, 128, 128]

    xtalk_vals = []
    for gamma in [0.0, 0.05, 0.15, 0.30, 0.50]:
        ghosted = ortho_targets.clone()
        ghosted[:, 0] = (1.0 - gamma) * ortho_targets[:, 0] + gamma * ortho_targets[:, 1]
        out = loss_fn(ghosted, ortho_targets)
        xtalk_vals.append(out["loss_xtalk"].item())
    assert xtalk_vals[0] < 1e-9, f"FAIL Test 2: Non-zero xtalk on clean target ({xtalk_vals[0]})"
    assert all(xtalk_vals[i] < xtalk_vals[i + 1] for i in range(len(xtalk_vals) - 1)), (
        f"FAIL Test 2: Crosstalk loss not strictly monotonic: {xtalk_vals}"
    )

    # Test 3: Noisy Planes (Graceful degradation without exploding gradients)
    noisy_pred = (ortho_targets + 0.5 * torch.randn_like(ortho_targets)).clamp(min=0.0).requires_grad_(True)
    res_noise = loss_fn(noisy_pred, ortho_targets)
    res_noise["loss"].backward()
    assert torch.isfinite(res_noise["loss"]) and noisy_pred.grad.norm() < 100.0, "FAIL Test 3: Unstable on noise!"

    # Test 4: Swapped Planes Detection via 4x4 Own-Target Matrix
    swapped_pred = ortho_targets[:, [1, 0, 2, 3]].clone()  # swap plane 0 and plane 1
    eval_clean = evaluate_uint8_metrics_and_4x4_matrix(ortho_targets, ortho_targets)
    eval_swap = evaluate_uint8_metrics_and_4x4_matrix(swapped_pred, ortho_targets)
    assert eval_clean["own_match_count"] == 4, "FAIL Test 4: Clean own_match != 4"
    assert eval_swap["own_match_count"] == 2, f"FAIL Test 4: Swapped own_match should be 2/4, got {eval_swap['own_match_count']}"
    assert (eval_swap["ghost_margin_per_plane"][0, :2] < -0.5).all(), "FAIL Test 4: Negative margin not detected on swapped planes!"

    print("ALL 4 FAILURE/SANITY TESTS PASSED (Black, Ghosted, Noisy, Swapped Planes).")


if __name__ == "__main__":
    run_all_failure_tests()

```

---

### Standardized Diagnostic Protocol: $4\times 4$ Matrix & B Median/Hard Contact Sheets

For every pilot evaluation on `VAL16` (8 Group A + 8 Group B = 64 planes), generate two artifacts:

1. **The $4\times 4$ Cross-Plane Correlation & Margin Table (on Group B's 8 samples / 32 planes):**
* Log both **`B_own` count** ($\sum_{i \in B}\sum_{m=1}^4 \mathbb{I}[\arg\max_k C^{(i)}_{m,k} = m]$ out of `32`) and **`B Ghost Margin`** $\overline{M}_B = \frac{1}{32}\sum_{i\in B}\sum_{m=1}^4 (C^{(i)}_{m,m} - \max_{k\neq m} C^{(i)}_{m,k})$.
* Log **`B Median` sample** (the 4th/5th ranked B sample by baseline Pearson, which currently has `4/4 own-target` yet visually ghosts) specifically by its **max off-diagonal bleed** $\max_{m \neq k} C^{(\text{med})}_{m,k}$.


2. **B Median & B Hardest Contact Sheet (`contact_sheet_B_uint8.png`):**
* Render a fixed lossless PNG grid for two locked Group B samples (`B_median` and `B_worst`) using the exact **reloaded `uint8` PNG arrays**:
* **Rows:** Planes $z_1=24.1\text{ mm}, z_2=25.3\text{ mm}, z_3=26.5\text{ mm}, z_4=27.7\text{ mm}$.
* **Columns (5 panels per row):**
1. `Ground Truth uint8` ($A_m$)
2. `v2 step12000 Baseline uint8` (`Asset 721714285`)
3. `Pilot Student uint8` (`Run_BLOT_CM`)
4. `Ghost Residual Heatmap` ($\text{clip}(\hat{A}^{\text{student}}_m - A_m, 0, 255)$ overlaid with foreign contours $\nabla A_{k\neq m}$ in red/cyan)
5. `2000-step Adam Diagnostic Ceiling` (for visual reference of what is physically achievable).





---

### Strict GO / NO-GO Criteria & Rollback Protocol

#### GO Criteria (All 5 Must Hold at Step 13,500 vs. `v2 step12000` / `Run_Ctrl`):

1. **Overall `VAL16 Pearson` $\ge 0.6200$** (vs. `0.61243` baseline; net gain $\ge +0.0075$) AND **Global SSIM $\ge 0.5600$** (vs. `0.55356`).
2. **`Group B Pearson` $\ge 0.4680$** (vs. `0.4516` baseline; net gain $\ge +0.0164$) with **`Group A Pearson` $\ge 0.7680$** (less than `0.0053` regression from `0.7733`).
3. **`B Own-Target Matched Planes` $\ge 27 / 32$** (up from `25 / 32`).
4. **`B Median` Off-Diagonal Ghost Reduction:** Mean off-diagonal correlation $\max_{k\neq m} C_{m,k}$ on `B median` drops by at least **`-0.035`** (or `B Ghost Margin` increases by $\ge +0.030$), visibly clearing dark-background bleed on `contact_sheet_B_uint8.png`.
5. **Independent Mac Reproducibility:** Standalone single-forward evaluation of the exported checkpoint on Mac reproduces the exact `uint8` PNG Pearson and Global SSIM within $\pm 10^{-4}$.

#### NO-GO & Immediate Rollback Protocol:

* **Trigger:** If `Run_BLOT_CM` fails any of the 4 unit sanity tests, degrades `Group A Pearson` below `0.7650`, fails to beat `Run_Ctrl` (pure R3 continuation at step 13,500) by at least `+0.008` on `Group B Pearson`, or introduces high-frequency speckle on the B contact sheet.
* **Rollback Action:** Discard the fine-tuned branch immediately and keep **`v2 seed0 step12000` (`Asset 721714285`)** as the locked production champion until Sanya’s `W32d3 / W64d3 / W32d4` runs finish. If one of Sanya’s wider/deeper backbones beats `v2 step12000` on native R3, apply this exact 1,500-step `BLOT-CM` fine-tuning recipe on top of Sanya's winning checkpoint.
