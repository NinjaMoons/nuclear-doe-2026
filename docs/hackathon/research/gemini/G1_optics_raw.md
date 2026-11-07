# G1 — raw Gemini answer (OPTICS), verbatim

- Prompt: "Сессия 1 — G1" block of `PROMPTS_SENT_G1_G2.md` (= GEMINI_TO_CLAUDE.md, LIVE FACTS + GEMINI 1).
- Provenance: pasted by the user (Grigoriy) into Claude Code chat ~00:00 MSK 08.11.2026; Gemini model/version: UNKNOWN (not stated).
- Text below is unchanged.

---

### 1. Scientifically Defensible 2-Paragraph Explanation

**Paragraph 1: Fundamental Optical Limits vs. Target-Phase Degrees of Freedom.**

In our exact geometry ($\lambda = 532\text{ nm}$, aperture $D = N\Delta x_0 = 1.28\text{ mm}$, $z_k \in \{24.1, 25.3, 26.5, 27.7\}\text{ mm}$, step $\Delta z = 1.2\text{ mm}$), a single thin phase-only DOE ($\vert{}e^{i\phi}\vert{} \equiv 1$) must satisfy Parseval energy conservation across all four planes simultaneously while operating at a modest numerical aperture ($\text{NA} \approx \frac{D}{2\bar{z}} \approx 0.0247$). Under the *paraxial local spatial-frequency approximation*, propagating a complex field from plane $z_k$ to $z_k + \Delta z$ multiplies each spatial frequency $\nu \in [0, 0.5]\text{ cycles/pixel}$ ($f = \nu / \Delta x_k$) by the Fresnel defocus transfer factor $H_{\Delta z}(\nu) = \exp\!\big(-i\pi \lambda \Delta z (\nu/\Delta x_k)^2\big) \approx \exp\!\big(-i \cdot 5.76\pi \nu^2 (\Delta z / 1.2\text{ mm})\big)$. Consequently, if the reconstructed field at plane $z_k$ has a **smooth or flat phase** $\psi_k(u,v) \approx \text{const}$, its bandwidth is restricted to the low spatial frequencies of the grayscale target envelope ($\nu \le 0.10\text{ cyc/px}$), where the axial phase shift is only $\Delta\Phi \le 0.058\pi\text{ rad}$ ($10.4^\circ$) per $1.2\text{ mm}$ step and $\le 0.173\pi\text{ rad}$ ($31^\circ$) across the entire $3.6\text{ mm}$ span—making low-frequency unmodulated amplitudes physically incapable of rapid axial decay. However, because the evaluation metric depends solely on **amplitude** $\vert{}U_k\vert{}$, the phase $\psi_k(u,v) = \arg U_k(u,v)$ inside bright target regions is a free variable: per-sample 2000-step Adam ($B\_P = 0.6526$) exploits this by injecting structured high-frequency phase variations ($\nu \in [0.25, 0.50]\text{ cyc/px}$, where $\Delta\Phi \in [0.36\pi, 1.44\pi]\text{ rad}$ per step) that constructively form $T_k$ at $z_k$ but diffract laterally across the maximum single-step ray cone $\Delta W = \frac{\lambda \Delta z}{\Delta x_0} = 63.8\text{ \mu m}$ ($\approx 6.1\text{ pixels}$ per $1.2\text{ mm}$, or $191.5\text{ \mu m} \approx 18.2\text{ pixels}$ across $3.6\text{ mm}$), diluting out-of-focus ghosts into a low-amplitude background halo.

**Paragraph 2: Why the 12ch STEM U-Net Exhibits Recoverable Axial Ghosting ($B\_P = 0.4516$ vs. $0.6526$).**

The gap between our 12ch STEM U-Net and the per-sample Adam reference is primarily a **recoverable neural representation and coordinate-coupling error**, driven by two mechanisms specific to the single-FFT architecture. First, the 8 analytic inverse-Fresnel STEM channels $\mathcal{P}_{z_k}^{-1}(T_k e^{i\cdot 0})$ are computed under a **zero-target-phase assumption** ($\psi_k \equiv 0$); feeding zero-phase fields into $\mathcal{P}_{z_k}^{-1}$ backprojects only the low-frequency stationary-phase envelope of $T_k$, which—as shown above—has near-zero axial selectivity and predisposes the network toward a smooth-phase compromise that persists across all four planes. Second, the single-FFT Fresnel transform maps DOE grid indices $(m,n)$ at fixed pitch $\Delta x_0 = 10.00\text{ \mu m}$ to target pixel indices $(p,q)$ with a **$z$-dependent physical pitch** $\Delta x_k = \frac{\lambda z_k}{N\Delta x_0} \in \{10.02, 10.52, 11.01, 11.51\}\text{ \mu m}$—a $14.94\%$ radial magnification from $z_1$ to $z_4$ (up to $95.8\text{ \mu m}$ or $\sim 9.1\text{ pixels}$ of physical shift at the sensor edge). While the 8 inverse-Fresnel channels live in the $10.00\text{ \mu m}$ DOE coordinate frame, the 4 raw target channels $T_1..T_4$ are concatenated directly by *pixel index* $(p,q)$ and passed through index-aligned U-Net skip connections; Combined with deterministic regression over a non-convex phase manifold (where infinitely many target phases $\psi_k$ yield the same $\vert{}U_k\vert{}$), the U-Net hedges by leaking index-aligned features directly across planes rather than committing to a specific high-frequency dispersive phase code.

---

### 2. Quantitative Optical Summary & Explicit Approximations

| Parameter / Regime | Exact Formula | Value in Our Setup ($\lambda=532\text{ nm}, N=128, \Delta x_0=10\text{ \mu m}$) | Physical / Modeling Implication |
| --- | --- | --- | --- |
| **Native Output Pitch $\Delta x_k$** | $\Delta x_k = \frac{\lambda z_k}{N \Delta x_0}$ | $10.02,\, 10.52,\, 11.01,\, 11.51\text{ \mu m}$ | Exact property of single-FFT Fresnel. $z_4/z_1 = 1.1494$ ($+14.94\%$ physical zoom). |
| **Edge Physical Drift** | $\Delta r_{\text{edge}} = \frac{N}{2}(\Delta x_4 - \Delta x_1)$ | $95.8\text{ \mu m}$ ($\approx 8.3\text{–}9.6\text{ px}$) | Raw target channels $T_1..T_4$ stacked by index $(p,q)$ are physically misaligned with $\mathcal{P}_{z_k}^{-1}(T_k)$. |
| **Max Diffraction Cone $\Delta W$** | $\Delta W(\Delta z) = 2 \lambda f_{0,\max} \Delta z = \frac{\lambda \Delta z}{\Delta x_0}$ | $63.8\text{ \mu m}$ ($\approx 6.1\text{ px}$) per $1.2\text{ mm}$<br>

<br>$191.5\text{ \mu m}$ ($\approx 18.2\text{ px}$) per $3.6\text{ mm}$ | *(Geometric/bandlimit bound)*: Max lateral radius over which Adam can disperse an out-of-focus feature. |
| **Axial Phase Shift $\Delta\Phi(\nu)$** | $\Delta\Phi \approx \pi \lambda \Delta z (\nu / \Delta x_k)^2$ | $\nu = 0.08\text{ cyc/px} \implies 0.037\pi\text{ rad}$<br>

<br>$\nu = 0.40\text{ cyc/px} \implies 0.92\pi\text{ rad}$ | *(Paraxial continuous-spectrum approx.)*: Smooth-phase low-$\nu$ features cannot dephase over $1.2\text{ mm}$; high-$\nu$ phase modulation dephases rapidly. |

---

### 3. Slide-Ready Diagram Proposal

**Slide Title:** *Anatomy of Group B Ghosting: Fundamental Low-$\nu$ Physics vs. Recoverable Neural Shortcuts*

* **Left Panel — Physical Propagation & Frequency Selectivity (The Optical Channel):**
* Draw the DOE plane ($D = 1.28\text{ mm}, \Delta x_0 = 10\text{ \mu m}$) propagating to $z_1..z_4$ ($24.1..27.7\text{ mm}$) as an expanding frustum showing the $+14.9\%$ physical grid expansion ($\Delta x_1 = 10.02\text{ \mu m} \to \Delta x_4 = 11.51\text{ \mu m}$).
* Show two wavepackets leaving a target feature at $z_2$:
1. *Red (Zero/Smooth Target Phase, $\nu < 0.1\text{ cyc/px}$):* Narrow beam ($\Delta\Phi < 0.06\pi$), stays collimated through $z_1, z_3, z_4$ $\to$ **Unavoidable low-frequency ghost if target phase is flat**.
2. *Green (Adam High-$\nu$ Target Phase Code, $\nu \in [0.25, 0.5]\text{ cyc/px}$):* Wide-angle bundle ($\Delta\Phi \sim \pi$), focuses sharply at $z_2$ and spreads across $\Delta W = \pm 32\text{ \mu m}$ ($\sim 6.1\text{ px}$ width) at $z_1, z_3$ $\to$ **Diluted background halo**.




* **Center Panel — Architectural Mismatch in Current 12ch STEM U-Net:**
* Show the two input branches entering the U-Net encoder:
* *Branch A (4 ch):* Raw $T_1..T_4$ in **Target Pixel Index $(p,q)$** (mismatched physical scale $\Delta x_k \propto z_k$, feeding skip connections directly to the output).
* *Branch B (8 ch):* $\mathcal{P}_{z_k}^{-1}(T_k e^{i\cdot 0})$ in **DOE Physical Coordinates $(m,n)$**, computed with **flat zero phase** ($\psi_k = 0$), which strips out high-$\nu$ dispersive trajectories.




* **Right Panel — Variance / Error Decomposition Bar (Conceptual Breakdown on Group B):**
* stacked horizontal bar from $B\_P = 0.0$ to $1.0$:
*   $[0.00 \to 0.45]$ **Current 12ch STEM v2 (`step12000`)**: Captures zero-phase stationary envelope + suffers index-shortcut bleed.
*   $[0.45 \to 0.65]$ **Recoverable Neural Gap (2000-step Adam Ceiling)**: Gained by synthesizing high-$\nu$ target-plane phase $\psi_k(u,v)$ and resolving coordinate scaling.
*   $[0.65 \to 1.00]$ **Physical Single-DOE Limit**: Finite aperture ($1.28\text{ mm}$), shared Parseval energy across 4 planes, and 8-bit quantization.



---

### 4. Two Unique, Falsifiable Read-Only Tests on Saved PNG / DOE

Both tests run strictly read-only in $<10\text{ seconds}$ on CPU using saved Group B artifacts: `Target PNGs`, `STEM v2 step12000 (DOE.png + recon PNGs)`, and `Adam-2000 (DOE.png + recon PNGs)`.

#### Test 1: Spatial-Frequency-Band Axial Selectivity & Target-Plane Phase Roughness Audit

* **Hypothesis Tested:** Per-sample Adam suppresses Group B ghosting not by violating low-frequency diffraction limits, but by injecting high-spatial-frequency phase modulation $\psi_k = \arg U_k$ inside bright target regions, whereas the 12ch STEM U-Net produces overly smooth target-plane phases and high low-frequency inter-$z$ bleed.
* **Equations:**
1. Partition the 2D Fourier plane of the reloaded `uint8` amplitude reconstructions $A_k \in [0, 1]^{128\times 128}$ into radial frequency bands $\nu = \frac{\sqrt{k_x^2 + k_y^2}}{128} \in [0, \frac{\sqrt{2}}{2}]$:
* **Low band ($B_{\text{low}}$):** $\nu \in (0,\, 0.08]\text{ cyc/px}$ (where $\Delta\Phi(\Delta z) < 0.04\pi$).
* **Mid/High band ($B_{\text{high}}$):** $\nu \in (0.08,\, 0.50]\text{ cyc/px}$ (where $\Delta\Phi(\Delta z) \in [0.04\pi, 1.44\pi]$).
Compute band-limited reconstructions $A_k^{\text{low}} = \mathcal{F}^{-1}\{M_{\text{low}} \mathcal{F}\{A_k\}\}$ and $A_k^{\text{high}} = \mathcal{F}^{-1}\{M_{\text{high}} \mathcal{F}\{A_k\}\}$, and measure band-specific **Inter-$z$ Pearson** $\bar{\rho}_{\text{inter}}^{\text{band}} = \frac{1}{6}\sum_{k<m} \text{Pearson}(A_k^{\text{band}}, A_m^{\text{band}})$ and **Own-Plane Pearson** $\bar{\rho}_{\text{own}}^{\text{band}} = \frac{1}{4}\sum_k \text{Pearson}(A_k^{\text{band}}, T_k^{\text{band}})$.


2. Propagate the saved `uint8` `DOE.png` through the official `single-FFT Fresnel` to obtain complex fields $U_k = \vert{}U_k\vert{} e^{i\psi_k}$ at $z_1..z_4$, and measure **Foreground Target-Plane Phase Roughness** inside bright target support $\Omega_k = \{T_k > Q_{0.75}(T_k)\}$:

$$R_\psi(k) = \frac{\sum_{(p,q)\in \Omega_k} \vert{}U_k(p,q)\vert{}^2 \left( \vert{}\text{wrap}(\Delta_p \psi_k)\vert{}^2 + \vert{}\text{wrap}(\Delta_q \psi_k)\vert{}^2 \right)}{2 \sum_{(p,q)\in \Omega_k} \vert{}U_k(p,q)\vert{}^2}$$




* **Expected Direction:**
* In $B_{\text{low}}$, both STEM U-Net and Adam-2000 will show elevated $\bar{\rho}_{\text{inter}}^{\text{low}}$ relative to targets (confirming the physical low-$\nu$ selectivity limit).
* In $B_{\text{high}}$, Adam-2000 will exhibit substantially lower $\bar{\rho}_{\text{inter}}^{\text{high}}$ and higher $\bar{\rho}_{\text{own}}^{\text{high}}$ than STEM v2, accompanied by a much larger foreground phase roughness $R_\psi$ in the target planes.


* **What Refutes the Story:** If Adam-2000 has the *same or lower* foreground phase roughness $R_\psi$ than STEM v2, or if the gap between STEM v2 and Adam-2000 is uniform across $B_{\text{low}}$ and $B_{\text{high}}$, then smooth target-phase collapsing in the zero-phase STEM is **not** the primary bottleneck.

#### Test 2: Pixel-Index Shortcut vs. Physically Aligned ($\Delta x_k \propto z_k$) & Optical Defocus Decomposition

* **Hypothesis Tested:** Distinguishes whether the ghost appearing in dark regions of plane $k$ from a bright feature in plane $m$ is caused by **(a) Neural Pixel-Index Bleed** at identical array indices $(p,q)$ (from stacking unscaled $T_1..T_4$ in the U-Net), or **(b) True Physical Fresnel Defocus** (which scales radially by $s_{m\to k} = z_m / z_k$ in pixel indices and blurs by $\mathcal{P}_{z_k}\mathcal{P}_{z_m}^{-1}$).
* **Equations:**
For each ordered pair $(k, m)$ ($k \neq m$) on Group B, define the exclusive ghost evaluation mask $M_{k \leftarrow m} = \{T_k \le Q_{0.25}(T_k)\} \cap \{r > 20\text{ px}\}$ (outer annulus where radial magnification shift $r\vert{}1 - z_m/z_k\vert{}$ is $1.5\text{ to }9.6\text{ pixels}$). Compare three candidate templates against the reloaded reconstruction $A_k$ inside $M_{k \leftarrow m}$:
1. **Index-Aligned Template (Neural Shortcut):** $G_{m\to k}^{\text{idx}}(p,q) = T_m(p,q)$ (zero radial scaling).
2. **Physically Aligned Template (Magnification Corrected):** $G_{m\to k}^{\text{phys}}(p,q) = T_m\!\left(\frac{z_k}{z_m}(p - 64) + 64,\, \frac{z_k}{z_m}(q - 64) + 64\right)$ (since physical coordinate $x = (p-64)\Delta x_k = (p'-64)\Delta x_m \implies p'-64 = \frac{z_k}{z_m}(p-64)$).
3. **Optical Defocus Template:** $G_{m\to k}^{\text{opt}}(p,q) = \big\vert{}\mathcal{P}_{z_k}\big(\mathcal{P}_{z_m}^{-1}(U_m)\big)\big\vert{}(p,q)$, where $U_m = \mathcal{P}_{z_m}(e^{i\phi_{\text{DOE}}})$.
Compute masked Pearson correlations $\rho_{\text{idx}}(k\leftarrow m) = \text{Pearson}_{M_{k\leftarrow m}}(A_k, G_{m\to k}^{\text{idx}})$ and $\rho_{\text{phys}}(k\leftarrow m) = \text{Pearson}_{M_{k\leftarrow m}}(A_k, G_{m\to k}^{\text{phys}})$, and their difference $\Delta\rho_{\text{shortcut}} = \rho_{\text{idx}} - \rho_{\text{phys}}$.


* **Expected Direction:**
* For **STEM U-Net v2 (`step12000`)**, $\rho_{\text{idx}} > \rho_{\text{phys}}$ ($\Delta\rho_{\text{shortcut}} > 0$) on outer-annulus features, proving that index-aligned U-Net skip connections stamp ghosts at fixed pixel indices $(p,q)$ in violation of physical Fresnel scaling.
* For **Adam-2000** (which has no U-Net skip connections and optimizes $\phi$ strictly through physical Fresnel operators), $\Delta\rho_{\text{shortcut}} \le 0$ (any residual crosstalk follows physical coordinates $\rho_{\text{phys}}$ / $\rho_{\text{opt}}$).


* **What Refutes the Story:** If $\rho_{\text{phys}} \ge \rho_{\text{idx}}$ for STEM U-Net v2 across the outer annulus ($r > 20\text{ px}$), then the raw 4-channel index shortcut is negligible and coordinate-warping the input/skip connections is unnecessary.

#### Minimal Read-Only Pseudocode for Tests 1 & 2

```python
import numpy as np
import torch
import torch.nn.functional as F
from doe_core.fresnel import propagate_fresnel_single_fft  # official propagator

Z_MM = [24.1, 25.3, 26.5, 27.7]


def band_split(img_4ch: torch.Tensor, nu_cut: float = 0.08):
  # img_4ch: (4, 128, 128) float32 reloaded from uint8 PNG (/255.0)
  fy = torch.fft.fftfreq(128, d=1.0)[:, None]
  fx = torch.fft.fftfreq(128, d=1.0)[None, :]
  nu = torch.sqrt(fx**2 + fy**2)
  spec = torch.fft.fft2(img_4ch)
  low = torch.fft.ifft2(spec * (nu <= nu_cut)).real
  high = torch.fft.ifft2(spec * (nu > nu_cut)).real
  return low, high


def warp_to_plane_k(img_m: torch.Tensor, z_k: float, z_m: float):
  # Maps physical coordinates of plane k onto pixel indices of plane m: scale = z_k / z_m
  scale = z_k / z_m
  theta = torch.tensor([[[scale, 0.0, 0.0], [0.0, scale, 0.0]]], dtype=torch.float32)
  grid = F.affine_grid(theta, size=(1, 1, 128, 128), align_corners=False)
  return F.grid_sample(img_m[None, None], grid, mode="bilinear", align_corners=False)[0, 0]


def run_diagnostics_on_quartet(T_4ch, A_4ch, doe_phase):
  # Test 1: Band-resolved inter-z & foreground target-phase roughness
  A_low, A_high = band_split(A_4ch, nu_cut=0.08)
  U_4ch = propagate_fresnel_single_fft(torch.exp(1j * doe_phase), Z_MM)  # (4, 128, 128) complex
  psi = torch.angle(U_4ch)
  dpx = torch.atan2(torch.sin(psi[:, :, 1:] - psi[:, :, :-1]), torch.cos(psi[:, :, 1:] - psi[:, :, :-1]))
  dpy = torch.atan2(torch.sin(psi[:, 1:, :] - psi[:, :-1, :]), torch.cos(psi[:, :-1, :] - psi[:, :-1, :]))  # wrapped diff
  # Compute weighted R_psi inside T_k > quantile(0.75) ...

  # Test 2: Index-aligned vs Physically-aligned ghost correlation in outer annulus (r > 20px)
  yy, xx = torch.meshgrid(torch.arange(128) - 64, torch.arange(128) - 64, indexing="ij")
  outer = (xx**2 + yy**2) > 20**2
  for k in range(4):
    dark_k = (T_4ch[k] <= torch.quantile(T_4ch[k], 0.25)) & outer
    for m in range(4):
      if k == m: continue
      T_m_phys = warp_to_plane_k(T_4ch[m], Z_MM[k], Z_MM[m])
      # Compare masked_pearson(A_4ch[k][dark_k], T_4ch[m][dark_k])
      #     vs masked_pearson(A_4ch[k][dark_k], T_m_phys[dark_k])

```

---

### 5. Single Scientifically Justified Model Improvement (Conditional on Tests 1 & 2)

If Test 1 confirms that the STEM U-Net suffers from smooth target-plane phases ($R_\psi^{\text{STEM}} \ll R_\psi^{\text{Adam}}$) and/or Test 2 confirms coordinate-scaling mismatch, **do not** just widen the U-Net (already tested by Sanya) and **do not** inject random/quadratic phase carriers (which scatter energy outside the finite $128^2$ sensor window).

Instead, implement **One-Step Closed-Loop Wirtinger Residual Refinement (`WirtingerRefinementHead`)** inside `model.forward()`:

1. **Mechanism:** Let the existing 12ch STEM U-Net produce an initial phase map $\phi^{(0)}(m,n)$. Inside the same forward pass, propagate $e^{i\phi^{(0)}}$ through the exact official `single-FFT Fresnel` to obtain complex fields $U_k^{(0)} = A_k^{(0)} e^{i\psi_k^{(0)}}$ at $z_1..z_4$.
2. **Exact Analytical Adjoint Backprojection:** Compute the analytical per-plane Wirtinger complex residual using the **true current optical phase** $\psi_k^{(0)}$ (not zero phase!) and backproject each plane to the DOE coordinate grid via the exact inverse Fresnel operator $\mathcal{P}_{z_k}^{-1}$:

$$E_k(m,n) = \mathcal{P}_{z_k}^{-1}\!\left[ \left(\tilde{A}_k^{(0)}(p,q) - T_k(p,q)\right) e^{i\psi_k^{(0)}(p,q)} \right](m,n), \quad k \in \{1..4\}$$


3. **Lightweight Correction Head:** Concatenate $\big[\cos\phi^{(0)}, \sin\phi^{(0)}, \operatorname{Re}(E_{1..4}), \operatorname{Im}(E_{1..4})\big]$ ($10\text{ channels}$, all strictly co-registered on the $10.00\text{ \mu m}$ DOE physical grid) into a tiny 3-layer $3\times 3$ Conv-GroupNorm-SiLU residual block ($<15\text{k}$ parameters) to predict the updated phase $\phi^{(1)} = \operatorname{wrap}(\phi^{(0)} + \Delta\phi)$.

**Why this is uniquely justified:**

* It directly fixes **both** physical flaws of the open-loop 12ch STEM: (a) $\mathcal{P}_{z_k}^{-1}$ now acts on the actual non-zero optical phase $e^{i\psi_k^{(0)}}$ rather than flat zero phase $e^{i\cdot 0}$, and (b) the residual $T_k - \tilde{A}_k^{(0)}$ is computed in native target indices $(p,q)$ and mapped back into physical DOE coordinates $(m,n)$ by $\mathcal{P}_{z_k}^{-1}$, automatically eliminating the $14.9\%$ physical pitch mismatch without lossy image interpolation.
* It requires **zero iterative loops at inference** (just 1 extra batched FFT/IFFT pair inside `forward()`, adding $<2\text{ ms}$ latency) and is orthogonal to Sanya's backbone width/depth sweep.
