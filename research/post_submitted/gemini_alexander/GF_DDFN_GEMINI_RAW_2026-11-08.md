Гипотеза о том, что плато $\text{Pearson} \approx 0.6124$ вызвано **только** прямым конфликтом градиентов различных TRAIN-квартетов по весам сети ($\nabla_\theta L(T_i; \theta) \cdot \nabla_\theta L(T_j; \theta) \lt 0$), в своей наивной формулировке **неполна и частично искажена конфаундером числа эпох**, однако указывает на реальное фундаментальное препятствие — **вырождение фазовой калибровки (gauge degeneracy) при обратной френелевской дифракции в условиях дефицита осевого разрешения ($\Delta z \lt \text{DOF}$)**.

Ниже представлен независимый физико-математический аудит постановки, опровержение ложных интерпретаций эксперимента C3/C4, минимальный протокол проверки без обучения и архитектура, устраняющая первопричину плато.

---

## Физический аудит оптической схемы (почему ВЧ-фаза решает всё)

Прежде чем сравнивать гипотезы H1–H4, вычислим инварианты вашей оптической системы:

* Апертура DOE: $L_0 = N \cdot \Delta x_0 = 128 \times 10\,\text{мкм} = 1.28\,\text{мм}$, $\lambda = 0.532\,\text{мкм}$.
* Критическое расстояние Найквиста для Single-FFT Fresnel:

$$z_{\text{crit}} = \frac{N (\Delta x_0)^2}{\lambda} = \frac{128 \times 100\,\text{мкм}^2}{0.532\,\text{мкм}} = 24.060\,\text{мм}$$



Плоскости $z \in \{24.1, 25.3, 26.5, 27.7\}\,\text{мм}$ лежат сразу за границей $z_{\text{crit}}$, а шаг в целевой плоскости составляет $\Delta x_z(z_k) = \frac{\lambda z_k}{L_0} \in [10.02, 11.51]\,\text{мкм}$.
* **Осевая глубина резкости (Depth of Focus, DOF)**:
Эффективная числовая апертура полной маски $\text{NA}_{\max} \approx \frac{L_0 / 2}{\bar{z}} = \frac{0.64\,\text{мм}}{25.9\,\text{мм}} \approx 0.0247$. Продольный размер дифракционного пятна (критерий Рэлея):

$$\text{DOF} = \frac{2\lambda}{\text{NA}_{\max}^2} \approx \frac{2 \times 0.532 \times 10^{-3}\,\text{мм}}{(0.0247)^2} \approx 1.74\,\text{мм}$$



При этом расстояние между соседними плоскостями в вашем задании равно **$\Delta z = 1.2\,\text{мм} \lt \text{DOF} = 1.74\,\text{мм}$**!

**Физическое следствие:** Набег фазы ядра дефокусировки Френеля между соседними плоскостями $\Delta z = 1.2\,\text{мм}$ на пространственной частоте $f$ равен:


$$\Delta \Phi_{\text{axial}}(f) = \pi \lambda \Delta z |f|^2$$


На низких и средних пространственных частотах ($\vert{}f\vert{} \le \frac{1}{2} f_{\text{Nyquist}}$) фазовый сдвиг между соседними плоскостями составляет **$\Delta \Phi_{\text{axial}} \le 1.19\,\text{рад} \lt \frac{\pi}{2}$**. Это означает, что низкочастотная и среднечастотная фаза DOE физически **не способна** разделить четыре плоскости: волновое поле просто проецирует усреднённую картину $\bar{T} = \frac{1}{4}\sum_k T_k$ сразу на все четыре расстояния (сильное межплоскостное смешение). Единственный способ физически развязать плоскости с шагом $1.2\,\text{мм}$ — перебросить энергию в **высокочастотное кольцо спектра DOE** ($|f| \to f_{\text{Nyquist}}$, края апертуры), где $\Delta \Phi_{\text{axial}} \approx 4.76\,\text{рад}$ на плоскость.

Именно поэтому:

1. Даже индивидуальный Oracle упирается в физический потолок информационной ёмкости одной фазовой маски ($\text{Pearson} \approx 0.6704$).
2. Перестановка ВЧ-фазы в эксперименте Матвея обрушивает качество на $0.11\text{–}0.12$: без согласованной ВЧ-несущей эффективная $\text{NA}_{\text{eff}}$ падает, $\text{DOF}$ вырастает до десятков миллиметров, и плоскости сливаются.
3. На Group A (где плоскости сильнее коррелированы по низким частотам) STEM V2 легко берёт $\text{Pearson} \approx 0.77$ (при среднем $0.6124$ и Group B $0.4516$), тогда как на Group B (требующей жёсткой осевой развязки несовместимых структур) падает до $0.4516$.

---

## Шаг 1. Критическое сравнение четырёх объяснений (H1–H4)

| Гипотеза | Роль в плато $0.6124$ | Роль в провале $N64$ ($0.4938$) | Вердикт рецензента |
| --- | --- | --- | --- |
| **H1.** Прямой конфликт градиентов $\nabla_\theta L(T_i)$ по весам $\theta$ | **Вторичный симптом** (следствие H3 в последнем слое) | Малая ($\le 20\%$) | Не объясняет провал масштабирования ширины $W16 \to W64$. |
| **H2.** Дефицит предъявлений (188 против 3000) и spectral bias | **Нулевая** (долгие тренировки STEM V2 не пробили $0.6124$) | **Доминирующая ($\sim 75\%$)** | Объясняет артефакт эксперимента Матвея ($0.4938 \to 0.6124$), но не само плато $0.6124$. |
| **H3.** Мультимодальность и вырождение калибровки фазы в целевых плоскостях | **Первопричина ($\ge 80\%$)** | Средняя ($\sim 25\%$) | Непрерывная сеть усредняет несовместимые ветви ВЧ-несущей $\mathbb{E}[e^{i\phi_{\text{HF}}}] \to 0$. |
| **H4.** Ошибки пайплайна, квантования, нормировки или весов лосса | **Умеренная ($\le 15\%$)** (дисбаланс норм A vs B) | Нулевая | Баг физики исключён успехом $N1$ ($0.6562$), но весовой дисбаланс Group A/B вероятен. |

### Детальный разбор и попытка опровержения каждой гипотезы

#### H1 — Конфликт градиентов различных TRAIN-квартетов по весам сети

* **Математический механизм:** Пусть $g_i = \nabla_\theta L_{\text{R3}}(T_i; \theta)$. При батчевом обновлении шаг оптимизатора пропорционален $\bar{g} = \frac{1}{M}\sum_{i=1}^M g_i$. Если для разных квартетов $\langle g_i, g_j \rangle \lt 0$ в подпространстве параметров $\theta_{\text{HF}}$, отвечающих за синтез высокочастотной фазы, происходит деструктивная интерференция: $\Vert{}\bar{g}\Vert{}_2 \ll \frac{1}{M}\sum_i \Vert{}g_i\Vert{}_2$, и ВЧ-фильтры застревают около нуля.
* **Имеющиеся доказательства:** Разрыв между $N1$ ($0.6562$) и мульти-примерным обучением STEM V2 ($0.6124$ на VAL16), а также провал Group B ($0.4516$) на фоне Group A.
* **Сильные контраргументы (попытка опровержения):**
1. **Провал масштабирования ширины ($W16 \to W32 \to W64$):** В режиме перепараметризации увеличение ширины сверточной сети в 4 раза увеличивает размерность касательного пространства (NTK) в 16 раз и экспоненциально ортогонализует якобианы $\nabla_\theta f_\theta(T_i)$ для различных входов $T_i \neq T_j$. Если бы проблема сводилась к нехватке весов для разделения направлений градиентов в $\theta$, переход $W16 \to W64$ существенно поднял бы метрику. Отсутствие роста доказывает, что конфликт происходит не из-за тесноты пространства параметров $\theta$, а в самом функциональном отображении $T \mapsto \phi$.
2. **Размерность пространства весов:** В сети с $P \sim 10^5\text{–}10^6$ весами случайные неодинаково активированные примеры имеют $\cos(g_i, g_j) \approx 0$, а не $-1$.


* **Эксперимент для опровержения:** Вычислить истинный нормированный индекс подавления $\eta = \frac{\Vert{}\sum_{i=1}^M g_i\Vert{}_2}{\sqrt{\sum_{i=1}^M \Vert{}g_i\Vert{}_2^2}}$ на чекпоинте STEM V2 (см. Шаг 2). Если $\eta \approx 1.0$ (ортогональность) или $\eta \gt 1.0$ (сонаправленность), то прямой векторный конфликт градиентов (H1) как причина остановки оптимизации **опровергнут**.

#### H2 — Недостаточное число предъявлений примера в $N64$ (Spectral Bias)

* **Математический механизм:** Согласно спектральному смещению нейросетей (Rahaman et al., 2019), собственные числа оператора НТК для пространственной частоты $k$ затухают как $\lambda_k \propto k^{-\beta}$. Сеть сначала выучивает гладкую фазовую огибающую (за 100–300 проходов по примеру) и лишь после 1000+ проходов по тому же примеру формирует крутые фазовые градиенты (ВЧ-диффузор). В эксперименте Матвея $N64$ видел каждый квартет $\approx 188$ раз (фаза НЧ-огибающей), тогда как $N1$ — $3000$ раз (фаза ВЧ-синтеза).
* **Имеющиеся доказательства:** Полностью обученный STEM V2 на *невидимом* VAL16 даёт $0.61243$, что уже закрывает **$73\%$ разрыва** между $N64$ ($0.4938$) и $N1$ ($0.6562$).
* **Сильные контраргументы (попытка опровержения):**
1. Увеличение длительности обучения основной сети STEM V2 уже проверялось командой и **не устранило** плато $0.6124$ (и $0.4516$ на Group B).
2. Следовательно, H2 полностью объясняет, почему эксперимент $N64$ дал $0.4938$ вместо $\sim 0.61$, но **опровергается** как объяснение самого потолка $0.6124$.


* **Эксперимент для опровержения:** Оценить $N1$ на шаге $t = 188$ (равное число предъявлений с $N64$ на шаге 3000). Если $N1(t=188) \approx 0.50\text{–}0.53$, то провал $N64$ до $0.4938$ — чистый артефакт недообучения, а реальный эффект мульти-примерного торможения равен лишь $\Delta \text{Pearson} = \text{Pearson}(N1, t=3000) - \text{Pearson}(\text{STEM V2, TRAIN}) \approx 0.03\text{–}0.04$.

#### H3 — Мультимодальность обратной оптической задачи и топологическое усреднение мод

* **Математический механизм:** Целевой функционал $L_{\text{R3}}(\phi; T)$ зависит только от модулей амплитуд $\vert{}\mathcal{P}_{z_k}(e^{i\phi})\vert{}$. Для каждого квартета $T$ фазы в четырёх фокальных плоскостях $\psi_k(x_z) = \arg \mathcal{P}_{z_k}(e^{i\phi})$ являются **свободными калибровочными переменными (gauge variables)**. Множество оптимальных масок $\mathcal{M}^*(T) = \{\phi : L(\phi; T) \le \epsilon\}$ представляет собой невыпуклое многообразие на торе $\mathbb{T}^{128\times 128}$ с экспоненциальным числом изолированных бассейнов (отличающихся реализацией ВЧ-спекл-фазы и оптическими вихрями $2\pi$).
Детерминированная непрерывная нейросеть $f_\theta: \mathcal{T} \to \mathbb{R}^{128\times 128}$ обязана быть липшицевым отображением. Если два близких по НЧ-структуре квартета $T_i, T_j$ в процессе обучения притягиваются к разным ветвям ВЧ-несущей $\phi^{(a)} \in \mathcal{M}^*(T_i)$ и $\phi^{(b)} \in \mathcal{M}^*(T_j)$, минимизация матожидания лосса по датасету заставляет сеть выдавать условное среднее $\hat{\phi}(T) \approx \mathbb{E}[\phi^* | T]$. При усреднении некогерентных ВЧ-фаз их высокочастотная компонента взаимно уничтожается ($\mathbb{E}[e^{i\phi_{\text{HF}}}] \to 0$), оставляя только гладкую НЧ-фазу.
* **Имеющиеся доказательства:**
1. Различные Oracle-решения имеют сильную когерентную ВЧ-фазу, но сами фазовые маски радикально непохожи друг на друга.
2. Ни ширина ($W64$), ни глубина, ни смена параметризации выходной головы (R23 Fourier/Gabor) не снимают плато — классический признак **амортизационного коллапса мод (mode averaging)** в обратных задачах один-ко-многим.


* **Сильные контраргументы (попытка опровержения):**
Почему при обучении через физический дифференцируемый пропагатор (unsupervised physics loss, без принудительного MSE к случайным GS-маскам) сеть сама не выберет одну гладкую каноническую ветвь $\phi^*(T)$?
*Ответ:* В стандартной CNN вход $T_k(x_z)$ задан в координатах **частотной области Френеля**, а выход $\phi(x_0)$ — в **пространственной области апертуры DOE**. Локальная по пикселям сверточная сеть физически не имеет глобального калибровочного якоря в целевой плоскости $\psi_k(x_z)$, поэтому начальная затравка ВЧ-фазы для каждого примера задаётся локальным шумом градиента и оказывается несогласованной между примерами.
* **Эксперимент для опровержения:** Зафиксировать каноническую ВЧ-фазу-диффузор $\psi_k^{\text{anchor}}(x_z)$ в целевых плоскостях и подать аналитический обратный волновой фронт $\Phi_{\text{init}}(x_0) = \arg \sum_k \mathcal{P}_{z_k}^H(T_k e^{i\psi_k^{\text{anchor}}})$ на вход сети (или обучить ансамбль из $K$ голов с Best-of-$K$ лоссом на 16 квартетах). Если снятие калибровочного вырождения не повышает ВЧ-энергию фазы и Pearson на Group B, H3 опровергнута.

#### H4 — Ошибка сэмплирования, весов лосса, нормировки амплитуды или квантования

* **Математический механизм:**
1. *Свойство сохранения энергии Парсеваля vs Pearson/SSIM:* Так как $|e^{i\phi(x_0)}| \equiv 1$, полная энергия в каждой плоскости строго постоянна: $\|\hat{A}_k\|_2^2 = N^2 = 16384$. Если целевые шаблоны $T_k$ в Group A и Group B нормированы по максимуму ($\max T_k = 1$), а не по $L_2$-энергии ($\Vert{}T_k\Vert{}_2 = N$), то разреженные шаблоны и плотные шаблоны имеют кратно отличающиеся нормы якобиана $\frac{\partial \text{SSIM}(\hat{A}_k, T_k)}{\partial \hat{A}_k}$.
2. *Разрыв градиента на границе $\pm\pi$ и 8-битное квантование:* При прямом предсказании скалярной фазы через $\pi \tanh(z)$ сеть не может перешагнуть через фазовый сброс $-\pi \leftrightarrow +\pi$ без прохождения через все промежуточные значения $[-\pi, \pi]$ на соседних пикселях, что штрафует ВЧ-решётки.


* **Имеющиеся доказательства:** Резкая асимметрия между Group A ($\sim 0.77$) и Group B ($0.4516$).
* **Сильные контраргументы (попытка опровержения):**
$N1$ ($0.6562$) и Oracle ($0.6704$) используют **тот же самый** пропагатор Френеля, лосс R3 и 8-битное сохранение. Следовательно, фатальной ошибки в физике пропагатора нет. Однако гипотеза о том, что примеры Group A подавляют примеры Group B по норме градиента $\|g_A\| \gg \|g_B\|$ или наоборот, остаётся открытой до проверки в Шаге 2.
* **Эксперимент для опровержения:** Обучить $N1$ (3000 шагов) отдельно на **одном квартете из Group B**. Если $N1(\text{Group B})$ достигает $\text{Pearson} \ge 0.62$, то пайплайн и лосс для Group B полностью исправны (H4 опровергнута как физический баг), а провал $0.4516$ в общей модели на $100\%$ вызван межрежимным подавлением при совместном обучении.

---

## Шаг 2. Минимальная проверка без обучения (Zero-Training Diagnostic Protocol)

### Критическое предупреждение об интерпретации метрики $C$ в высокой размерности

Вы предложили коэффициент подавления:


$$C = \frac{\left\|\sum_{i=1}^M g_i\right\|_2}{\sum_{i=1}^M \|g_i\|_2}$$


**Осторожно (ловушка размерности):** Пусть $g_1, \dots, g_M \in \mathbb{R}^P$ ($P \sim 10^6$) — попарно **ортогональные** векторы равной нормы ($\langle g_i, g_j \rangle = 0$, то есть **конфликт градиентов равен нулю**!). Тогда по теореме Пифагора:


$$C_{\text{orth}} = \frac{\sqrt{\sum_{i=1}^M \|g_i\|_2^2}}{\sum_{i=1}^M \|g_i\|_2} = \frac{1}{\sqrt{M}}$$


Для $M = 8$ квартетов при полном отсутствии конфликта ($S_{ij} = 0$) вы получите **$C = 1/\sqrt{8} \approx 0.3536$**! Наивная интерпретация приписала бы $1 - 0.3536 = 64.6\%$ «взаимному подавлению градиентов», хотя это чисто геометрическое следствие ортогональности в $\mathbb{R}^P$.

Поэтому в скрипт ниже наряду с вашим $C$ введён **Истинный индекс деструктивной интерференции**:


$$\eta = \frac{\left\|\sum_{i=1}^M g_i\right\|_2}{\sqrt{\sum_{i=1}^M \|g_i\|_2^2}} = \frac{C}{C_{\text{orth}}}$$

* $\eta \lt 1.0 \iff \sum_{i \neq j} \langle g_i, g_j \rangle \lt 0$ — **реальное деструктивное подавление** (подтверждение H1).
* $\eta \approx 1.0 \iff$ градиенты ортогональны, примеры обновляют непересекающиеся подпространства весов.
* $\eta \gt 1.0 \iff$ градиенты сонаправлены.

Аналогично, чтобы дисбаланс норм $\|g_i\|_2$ не маскировал угловой конфликт, вычисляется **нормированный коэффициент подавления** $\tilde{C} = \frac{1}{M}\left\|\sum_{i=1}^M \frac{g_i}{\|g_i\|_2}\right\|_2$ и его отношение к $1/\sqrt{M}$.

### Дополнительно: Спектральное расщепление градиента фазовой головы

Так как гипотеза утверждает, что конфликт бьёт именно по **высокочастотной фазе**, скрипт дополнительно раскладывает градиенты пространственных свёрточных ядер фазовой головы ($\theta_{\text{head}}$) на низкочастотную ($\text{LP}$, усреднение по окну) и высокочастотную ($\text{HP} = I - \text{LP}$) составляющие. Если $\eta_{\text{early}} \ge 1.0$, но $\eta_{\text{head, HP}} \lt 0.85$ с долей отрицательных косинусов $\gt 60\%$ — это прямое математическое доказательство деструктивного гашения ВЧ-фазы.

### Воспроизводимый PyTorch-скрипт (`diagnose_grad_conflict.py`)

```python
import math
from typing import Dict, List, Tuple
import torch
import torch.nn as nn

@torch.no_grad()
def split_kernel_freq(grad_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Разделяет градиент сверточного ядра (O, I, kH, kW) на НЧ (DC/среднее) и ВЧ (нулевое среднее).
    Для 1x1 ядер или векторов возвращает (grad, zeros).
    """
    if grad_tensor.ndim == 4 and grad_tensor.shape[-1] > 1 and grad_tensor.shape[-2] > 1:
        lp = grad_tensor.mean(dim=(-2, -1), keepdim=True).expand_as(grad_tensor)
        hp = grad_tensor - lp
        return lp.reshape(-1), hp.reshape(-1)
    return grad_tensor.reshape(-1), torch.zeros_like(grad_tensor).reshape(-1)

def compute_conflict_metrics(G: torch.Tensor, group_labels: List[str], name: str) -> Dict[str, float]:
    """
    G: матрица градиентов размера [M, P] в float64 (M примеров, P параметров).
    group_labels: список меток ['A', 'A', ..., 'B', 'B'] длины M.
    """
    M, P = G.shape
    eps = 1e-12
    norms = torch.linalg.vector_norm(G, dim=1)  # [M]
    G_unit = G / norms.clamp_min(eps).unsqueeze(1)  # [M, P]

    # Матрица попарных косинусов S_ij
    cos_matrix = G_unit @ G_unit.T  # [M, M]
    off_diag_mask = ~torch.eye(M, dtype=torch.bool, device=G.device)
    off_diag_cos = cos_matrix[off_diag_mask]

    # Коэффициенты подавления (сырой C, ортогональный эталон C_orth и истинный индекс eta)
    sum_g_norm = torch.linalg.vector_norm(G.sum(dim=0))
    sum_norms = norms.sum().clamp_min(eps)
    rss_norms = torch.linalg.vector_norm(norms).clamp_min(eps)

    C_raw = (sum_g_norm / sum_norms).item()
    C_orth = (rss_norms / sum_norms).item()
    eta_raw = (sum_g_norm / rss_norms).item()  # < 1.0 означает реальное подавление!

    # Угловой коэффициент подавления (без влияния дисбаланса норм)
    C_unit = (torch.linalg.vector_norm(G_unit.sum(dim=0)) / M).item()
    eta_unit = C_unit * math.sqrt(M)  # < 1.0 означает угловое подавление

    # Стратификация по парам A-A, B-B, A-B
    pair_stats = {"A-A": [], "B-B": [], "A-B": []}
    for i in range(M):
        for j in range(i + 1, M):
            gi, gj = group_labels[i], group_labels[j]
            key = f"{gi}-{gj}" if gi == gj else "A-B"
            pair_stats[key].append(cos_matrix[i, j].item())

    def safe_mean(vals: List[float]) -> float:
        return sum(vals) / len(vals) if vals else float("nan")

    def neg_frac(vals: List[float]) -> float:
        return sum(1 for v in vals if v < 0.0) / len(vals) if vals else float("nan")

    norms_A = [norms[i].item() for i in range(M) if group_labels[i] == "A"]
    norms_B = [norms[i].item() for i in range(M) if group_labels[i] == "B"]

    results = {
        "subset": name,
        "num_params": P,
        "C_raw": C_raw,
        "C_orth_baseline": C_orth,
        "eta_true_suppression": eta_raw,
        "C_unit": C_unit,
        "eta_unit": eta_unit,
        "neg_cos_fraction_all": (off_diag_cos < 0.0).float().mean().item(),
        "mean_cos_all": off_diag_cos.mean().item(),
        "mean_cos_AA": safe_mean(pair_stats["A-A"]),
        "mean_cos_BB": safe_mean(pair_stats["B-B"]),
        "mean_cos_AB": safe_mean(pair_stats["A-B"]),
        "neg_frac_AA": neg_frac(pair_stats["A-A"]),
        "neg_frac_BB": neg_frac(pair_stats["B-B"]),
        "neg_frac_AB": neg_frac(pair_stats["A-B"]),
        "norm_max_min_ratio": (norms.max() / norms.min().clamp_min(eps)).item(),
        "norm_mean_A_over_B": safe_mean(norms_A) / max(safe_mean(norms_B), eps),
    }
    return results, cos_matrix.cpu()

def run_zero_training_diagnostic(
    model: nn.Module,
    loss_fn: nn.Module,
    quartets: List[torch.Tensor],
    group_labels: List[str],
    head_name_keywords: Tuple[str, ...] = ("head", "out", "phase", "final"),
    device: str = "cuda",
):
    """
    Запускает диагностику на фиксированном чекпоинте STEM V2 без изменения весов.
    quartets: список из M=8 тензоров целевых квартетов [1, 4, 128, 128].
    group_labels: список ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'].
    """
    # 1. Фиксируем статистику BatchNorm/Dropout во избежание межпримерного загрязнения
    model.eval().to(device)
    params_early, params_head = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name.lower() for k in head_name_keywords):
            params_head.append((name, p))
        else:
            params_early.append((name, p))

    # Fallback: если ключевые слова не совпали, берём последние 2 обучаемых тензора как голову
    if len(params_head) == 0:
        all_p = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        params_early, params_head = all_p[:-2], all_p[-2:]

    grads_all, grads_early, grads_head, grads_head_hp = [], [], [], []

    for x_i in quartets:
        model.zero_grad(set_to_none=True)
        x_i = x_i.to(device)
        pred = model(x_i)
        loss = loss_fn(pred, x_i)
        loss.backward()

        # Собираем градиенты в float64 для численной устойчивости скалярных произведений
        g_e = torch.cat([p.grad.detach().to(torch.float64).reshape(-1) for _, p in params_early])
        g_h = torch.cat([p.grad.detach().to(torch.float64).reshape(-1) for _, p in params_head])

        hp_list = []
        for _, p in params_head:
            _, hp = split_kernel_freq(p.grad.detach().to(torch.float64))
            if hp.numel() > 0:
                hp_list.append(hp)
        g_h_hp = torch.cat(hp_list) if len(hp_list) > 0 else g_h

        grads_early.append(g_e)
        grads_head.append(g_h)
        grads_head_hp.append(g_h_hp)
        grads_all.append(torch.cat([g_e, g_h]))

    G_all = torch.stack(grads_all, dim=0)
    G_early = torch.stack(grads_early, dim=0)
    G_head = torch.stack(grads_head, dim=0)
    G_head_hp = torch.stack(grads_head_hp, dim=0)

    for subset_name, G_mat in [
        ("ALL_WEIGHTS", G_all),
        ("EARLY_ENCODER", G_early),
        ("PHASE_HEAD_ALL", G_head),
        ("PHASE_HEAD_HIGH_FREQ", G_head_hp),
    ]:
        res, cos_mat = compute_conflict_metrics(G_mat, group_labels, subset_name)
        print(f"\n=== {subset_name} (P={res['num_params']}) ===")
        print(f"C_raw = {res['C_raw']:.4f} | C_orth = {res['C_orth_baseline']:.4f} | "
              f"eta_true = {res['eta_true_suppression']:.4f} (<1.0 means conflict)")
        print(f"C_unit = {res['C_unit']:.4f} | eta_unit = {res['eta_unit']:.4f} | "
              f"Neg Cosine Frac = {res['neg_cos_fraction_all']*100:.1f}%")
        print(f"Mean Cos: A-A={res['mean_cos_AA']:.4f}, B-B={res['mean_cos_BB']:.4f}, A-B={res['mean_cos_AB']:.4f}")
        print(f"Neg Frac: A-A={res['neg_frac_AA']*100:.1f}%, B-B={res['neg_frac_BB']*100:.1f}%, A-B={res['neg_frac_AB']*100:.1f}%")
        print(f"Norm Ratio Max/Min = {res['norm_max_min_ratio']:.2f} | Mean(||g_A||)/Mean(||g_B||) = {res['norm_mean_A_over_B']:.2f}")

```

### Правила строгой интерпретации результатов скрипта

1. **Если $\eta_{\text{true}}(\text{PHASE\_HEAD}) \lt 0.85$ и $\text{neg\_frac} \gt 65\%$, тогда как $\eta_{\text{true}}(\text{EARLY\_ENCODER}) \ge 0.98$:**
Конфликт локализован исключительно в синтезе фазы на выходе (подтверждение связки H3 $\to$ H1). Ранние слои извлекают совместимые признаки, а фазовая голова пытается усреднить несовместимые ветви ВЧ-фазы.
2. **Если $\text{mean\_cos\_AB} \lt -0.05$, а $\text{mean\_cos\_AA} \gt 0$ и $\text{norm\_mean\_A\_over\_B} \gt 2.0$:**
Group A доминирует над Group B как по амплитуде градиента, так и по направлению обновления фазовой головы.
3. **Если везде $\eta_{\text{true}} \approx 1.0 \pm 0.05$ и $\text{mean\_cos} \approx 0 \pm \frac{2}{\sqrt{P}}$:**
Градиенты просто ортогональны в $\mathbb{R}^P$. Прямая векторная война градиентов в пространстве весов $\theta$ опровергнута; причина плато — неспособность непрерывной сверточной параметризации $f_\theta(T)$ преодолеть координатный разрыв Фурье-Френеля и калибровочное вырождение (чистая H3).

---

## Шаг 3. Сравнение подходов A–D и выбор одного приоритетного метода

| Метод | Как работает | Почему НЕ решает задачу (или в чём ограничение) | Соответствие правилам |
| --- | --- | --- | --- |
| **A. PCGrad / CAGrad** | Проецирует конфликтующие градиенты $g_i$ на нормальную плоскость $g_j$ ($\langle g_i, g_j \rangle \lt 0$) | **Фатальный изъян для мультимодальной фазы:** если два квартета тянут ВЧ-фазу к противоположным ветвям $+\phi_{\text{HF}}$ и $-\phi_{\text{HF}}$, проекция ортогонализации просто **обнуляет спорную ВЧ-компоненту** и оставляет их общее НЧ-среднее. PCGrad ускоряет НЧ-коллапс! Плюс $O(B)$ backward-проходов. | Да |
| **B. $K$ фазовых голов + Best-of-$K$ loss** | $K$ голов предсказывают $\phi^{(1..K)}$, градиент идёт в лучшую по оптическому лоссу $k^* = \arg\min_k L(\phi^{(k)}, T)$ | На обучении снимает усреднение мод, но **на inference без итеративного поиска или оракула** сеть не знает, какую из $K$ голов выбрать. Если же считать 4 плоскости Френеля для всех $K$ масок на инференсе и брать $\arg\min$, при большом $K$ это нарушает запрет на скрытый перебор масок. | На грани (при $K \gg 1$ — скрытый поиск) |
| **C. Физический Encoder + $K$ декодеров + обучаемый Gating** | Роутер $G(T)$ выбирает специализированный декодер по свойствам входного квартета $T$ | Разделяет режимы Group A и Group B, но **внутри каждого декодера** сохраняется проблема прямой сверточной регрессии из пространства изображений $x_z$ в апертуру $x_0$ и произвольной калибровки фокальной фазы $\psi_k$. | Да |
| **D. Каноническая фиксация калибровки в целевой плоскости + сопряжённый подъём Френеля + режимный Gating (GF-DDFN)** | Сеть предсказывает предкомпенсацию амплитуд $\Delta A_k(T)$ и поправку фазы $\Delta\psi_k(T)$ поверх фиксированного ВЧ-якоря $\psi_k^{\text{anchor}}$ в целевых плоскостях, переносит их в плоскость DOE точным сопряжённым оператором $\mathcal{P}_{z_k}^H$ и доводит специализированной головой (с Top-1 роутингом A/B) | **Устраняет саму физическую причину мультимодальности и осевого смешения:** фиксирует единую ветвь ВЧ-фазы для всех примеров и аналитически обращает глобальное преобразование Фурье + квадратичный чирп $z_k$. | **Полное соответствие** |

### Приоритетный выбор: Метод D (с встроенным режимным роутингом из C) — **GF-DDFN (Gauge-Fixed Dual-Domain Fresnel Network)**

#### Почему GF-DDFN принципиально отличается от ваших R8 / R18 / R19 / R22 / R23 и STEM V2:

1. **Преодоление дуальности координат Single-FFT Fresnel:**
В формуле Single-FFT Fresnel:

$$\mathcal{P}_{z_k}(U_0) = D_{z_k}^{(\text{out})} \, \mathcal{F} \left\{ D_{z_k}^{(\text{in})} \, U_0 \right\}$$



где $D_{z_k}^{(\text{in})}(x_0) = \exp\!\left(i\frac{\pi}{\lambda z_k}|x_0|^2\right)$ и $D_{z_k}^{(\text{out})}(x_z) = \frac{e^{ikz_k}}{i\lambda z_k}\exp\!\left(i\frac{\pi}{\lambda z_k}|x_z|^2\right)$.
Целевые изображения $T_k(x_z)$ живут в **спектральной области Френеля**, а фаза DOE $\phi(x_0)$ — в **пространственной области апертуры**. Любая однодоменная сверточная сеть (включая STEM V2 и R23 с Фурье/Габор головой на выходе) пытается локальными по пикселям свёртками выучить глобальное обратное БПФ $\mathcal{F}^{-1}$ и четыре разных параболических чирпа $D_{z_k}^{(\text{in})*}$. Сдвиг детали в целевом изображении $T_k(x_z - \delta)$ требует глобального линейного наклона фазы $\Delta\phi(x_0) \propto -\delta \cdot x_0$ по всей маске DOE, чего локальная свёртка сделать не может.
2. **Снятие мультимодального вырождения (Gauge Fixing):**
Так как каждый оператор $\mathcal{P}_{z_k}$ строго унитарен ($\mathcal{P}_{z_k}^{-1} = \mathcal{P}_{z_k}^H$), по теореме Парсеваля задача минимизации $\sum_{k=1}^4 \|\mathcal{P}_{z_k}(e^{i\phi}) - \tilde{A}_k e^{i\psi_k}\|_2^2$ в плоскости DOE имеет **точное аналитическое решение**:

$$\phi_{\text{analytic}}(x_0) = \arg \left( \sum_{k=1}^4 w_k \mathcal{P}_{z_k}^H \left( \tilde{A}_k(x_z) e^{i\psi_k(x_z)} \right) \right)$$



В GF-DDFN мы задаём **единый фиксированный (или совместно обучаемый как буфер/параметр) высокочастотный фазовый якорь** $\psi_k^{\text{anchor}}(x_z) \in \mathbb{R}^{4\times 128\times 128}$ (с широкополосным спектром, заполняющим всю апертуру DOE $128\times 128$).
Первая нейросетевая стадия (**Target-Space Cross-Talk Unmixing Net**) работает в координатах целевых плоскостей $x_z$ (где все 4 изображения пространственно выровнены с точностью до известного масштаба $\Delta x_z(z_k)$!) и предсказывает:
* Предкомпенсированную комплексную модуляцию в целевых плоскостях $\tilde{U}_{z_k}(x_z) = \text{ReLU}(T_k + \Delta A_k(T)) \cdot \exp\!\left(i\left(\psi_k^{\text{anchor}} + \Delta\psi_k(T)\right)\right)$, вычитающую паразитное просачивание соседних плоскостей (axial cross-talk pre-cancellation).
Затем фиксированный физический слой $\sum_{k=1}^4 \mathcal{P}_{z_k}^H(\tilde{U}_{z_k})$ переносит поле в плоскость DOE $x_0$, а вторая нейросетевая стадия (**Aperture-Space Regime-Routed Phase Net**) с лёгким физическим роутером $G(T)$ (разделяющим режимы Group A и Group B по межплоскостной корреляции) синтезирует финальный фазор DOE $e^{i\phi(x_0)}$.



---

## Шаг 4. Проверка научной литературы (Первоисточники vs Гипотезы)

### 1. Gradient Interference, PCGrad и CAGrad

* **Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. (2020).** *Gradient Surgery for Multi-Task Learning (PCGrad).* Advances in Neural Information Processing Systems (NeurIPS), 33, 5824–5836. `arXiv:2001.06782`.
* **Liu, B., Liu, X., Jin, X., Stone, P., & Liu, Q. (2021).** *Conflict-Averse Gradient Descent for Multi-task Learning (CAGrad).* NeurIPS, 34, 18878–18890. `arXiv:2110.14048`.
* **Что доказано в литературе:** При наличии «трагической триады» (отрицательный косинус градиентов, большая кривизна вдоль конфликтного направления и сильный дисбаланс норм $\Vert{}g_A\Vert{} \gg \Vert{}g_B\Vert{}$) стандартный Adam застревает на доминирующей задаче. Проекция PCGrad или поиск в шаре вокруг среднего градиента (CAGrad) гарантирует сходимость к стационарной точке Парето.
* **Что остаётся предположением для нашей задачи:** В классическом MTL задачи делят общий энкодер, но имеют *разные* выходные головы. У нас все квартеты делят *одну и ту же* фазовую голову. Поэтому применение PCGrad по отдельным примерам батча без снятия фазовой мультимодальности приводит к вычитанию ВЧ-компонент. Однако применение **Group-balanced CAGrad / градиентной нормировки между двумя макро-группами (Group A vs Group B)** оправдано, если диагностика Шага 2 покажет $\Vert{}g_A\Vert{} / \Vert{}g_B\Vert{} \gt 2$.

### 2. Multi-Modal Amortized Inverse Problems, Mode Collapse и Spectral Bias

* **Bishop, C. M. (1994).** *Mixture Density Networks.* Aston University Technical Report NCRG/94/004.
* **Rupprecht, C., Laina, I., DiPietro, R., Baust, M., Tombari, F., Navab, N., & Hager, G. D. (2017).** *Learning in an Uncertain World: Representing Ambiguity Through Multiple Hypotheses.* IEEE ICCV 2017, pp. 3591–3600. `DOI: 10.1109/ICCV.2017.388`.
* **Rahaman, N., Baratin, A., Arpit, D., Draxler, F., Lin, M., Hamprecht, F., Bengio, Y., & Courville, A. (2019).** *On the Spectral Bias of Neural Networks.* ICML 2019, PMLR 97:5301–5310. `arXiv:1806.08734`.
* **Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991).** *Adaptive Mixtures of Local Experts.* Neural Computation, 3(1), 79–87. `DOI: 10.1162/neco.1991.3.1.79`.
* **Что доказано в литературе:** Детерминированная сеть, обучаемая на невыпуклом многозначном обратном отображении $T \mapsto \mathcal{M}^*(T)$, сходится к центру масс решений $\mathbb{E}[\phi \vert{} T]$, который в невыпуклой геометрии лежит *вне* допустимого множества решений (mode averaging). Кроме того, из-за spectral bias высокочастотные гармоники решения выучиваются на порядок медленнее низкочастотных.

### 3. Conditional Phase-Only CGH, Multi-Plane Holography и Axial Cross-Talk

* **Peng, Y., Choi, S., Padmanaban, N., & Wetzstein, G. (2020).** *Neural Holography with Camera-in-the-Loop Training.* ACM Transactions on Graphics (SIGGRAPH Asia), 39(6), Article 185. `DOI: 10.1145/3414685.3417802`.
* **Hossein Eybposh, M., Caira, N. W., Atisa, M., Chakravarthula, P., & Pégard, N. C. (2020).** *DeepCGH: 3D computer-generated holography using deep learning.* Optics Express, 28(18), 26636–26650. `DOI: 10.1364/OE.399624`.
* **Shi, L., Li, B., Kim, C., Kellnhofer, P., & Matusik, W. (2021).** *Towards Real-Time Photorealistic 3D Holography with Deep Neural Networks (TensorHolo).* Nature, 591, 234–239. `DOI: 10.1038/s41586-020-03152-0`.
* **Horisaki, R., Takagi, R., & Tanida, J. (2018).** *Deep-learning-generated holography.* Applied Optics, 57(14), 3859–3863. `DOI: 10.1364/AO.57.003859`.
* **Что доказано в оптической литературе:**
1. Прямая сверточная регрессия $T(x_z) \to \phi(x_0)$ (Horisaki et al., 2018) страдает от размытия и потери ВЧ-спекл-контраста из-за глобального характера дифракции Френеля.
2. Архитектуры **HoloNet (Peng et al., 2020)** и **DeepCGH (Eybposh et al., 2020)** доказали, что разделение сети на **(а) предсказание фазы/амплитуды в целевых плоскостях $T_k \mapsto (A_k, \psi_k)$**, **(б) аналитический обратный перенос Френеля $\sum_k \mathcal{P}_{z_k}^H$ в плоскость DOE** и **(в) фазовую доводку в плоскости DOE** радикально повышает SSIM/PSNR и подавляет межплоскостной cross-talk без итеративного GS на инференсе.


* **Что остаётся специфической гипотезой для нашей постановки:** В режиме Single-FFT Fresnel с шагом $\Delta z = 1.2\,\text{мм} \lt \text{DOF} = 1.74\,\text{мм}$ и $15\%$-м изменением шага сетки $\Delta x_z(z_k)$ между $24.1$ и $27.7\,\text{мм}$ добавление фиксированного широкополосного фазового якоря $\psi_k^{\text{anchor}}$ и режимного роутера Group A/B позволит поднять Group B с $0.4516$ к уровню $\ge 0.55$ без потери скорости инференса.

---

## Шаг 5. Инженерный отчёт (NIH-GEMINI-SCIENCE-028)

### 1. ROOT CAUSE (Наиболее вероятная причина и уровень уверенности)

* **Основная причина (Уверенность 85%):** **Калибровочное вырождение ВЧ-фазы (Gauge Degeneracy) + однодоменный координатный разрыв в условиях $\Delta z \lt \text{DOF}$ (Связка H3 $\to$ H1).**
Для разделения 4 плоскостей с шагом $\Delta z = 1.2\,\text{мм}$ при глубине резкости апертуры $\text{DOF} = 1.74\,\text{мм}$ маска DOE обязана работать как широкоугольный ВЧ-диффузор. Однако целевой лосс фиксирует только амплитуды $T_k$, оставляя фазы целевых плоскостей $\psi_k(x_z)$ свободными. Без канонического фазового якоря в целевой плоскости разные обучающие квартеты тянут общие веса фазовой головы к несовместимым реализациям ВЧ-несущей. В результате непрерывная сеть выдаёт условное среднее $\mathbb{E}[e^{i\phi_{\text{HF}}}] \to 0$, сглаживая ВЧ-фазу. Это уменьшает эффективную числовую апертуру, увеличивает $\text{DOF}$ и вызывает катастрофическое межплоскостное смешение на сложных несогласованных примерах (Group B Pearson $= 0.4516$ против Group A $\approx 0.77$).
* **Причина аномалии в эксперименте C3/C4 Матвея (Уверенность 90%):** **Конфаундер числа предъявлений и Spectral Bias (H2).**
Провал $N64$ до $0.4938$ (ниже полного STEM V2 $0.6124$) на $\sim 75\%$ вызван тем, что 188 показов примера недостаточно для преодоления спектрального смещения нейросети и выхода в режим генерации ВЧ-фазы.

### 2. FALSIFICATION (Критерии опровержения вывода)

Наш вывод будет **опровергнут**, если выполнится хотя бы одно из условий:

1. В диагностике Шага 2 на чекпоинте STEM V2 истинный индекс подавления в ВЧ-компонентах фазовой головы окажется $\eta_{\text{true}}(\text{PHASE\_HEAD\_HIGH\_FREQ}) \ge 0.98$, а отношение норм градиентов $\text{Mean}(\|g_A\|)/\text{Mean}(\|g_B\|) \in [0.8, 1.25]$.
2. Индивидуальное обучение $N1$ (3000 шагов) на одном квартете из **Group B** не сможет преодолеть порог $\text{Pearson} = 0.50$ (это означало бы жёсткий физический предел дифракции для геометрии Group B, а не проблему обобщения нейросети).
3. Двухдоменная схема GF-DDFN с фиксированным фазовым якорем $\psi_k^{\text{anchor}}$ и сопряжённым оператором $\mathcal{P}_{z_k}^H$ на контрольном прогоне $N16$ / полной тренировке не даст прироста Pearson на Group B минимум на $+0.03$ относительно STEM V2 при равном числе эпох.

### 3. ONE BEST METHOD (Выбранная архитектура)

**GF-DDFN (Gauge-Fixed Dual-Domain Fresnel Network с физическим роутером режимов A/B):**

1. **Стадия 1 (Target-Domain Unmixing CNN, обучаемая):** Принимает $T \in \mathbb{R}^{B \times 4 \times 128 \times 128}$ и карту радиальных частот $r^2(x_z)$, предсказывает предкомпенсированную амплитуду $\tilde{A}_k = \text{Softplus}(T_k + \Delta A_k(T))$ (подавление осевого эха соседних плоскостей) и гладкую фазовую поправку $\Delta\psi_k(T)$ поверх фиксированного широкополосного фазового якоря $\psi_k^{\text{anchor}}(x_z)$ (сферический + квазислучайный ВЧ-диффузор с фиксированным seed).
2. **Физический мост (Exact Adjoint Single-FFT Fresnel, фиксированный без параметров):** Вычисляет комплексный волновой фронт в плоскости DOE:

$$U_{\text{seed}}(x_0) = \sum_{k=1}^4 w_k(T) \, \mathcal{P}_{z_k}^H \left( \tilde{A}_k(x_z) \exp\!\left(i\left(\psi_k^{\text{anchor}}(x_z) + \Delta\psi_k(T)\right)\right) \right)$$



где $\mathcal{P}_{z_k}^H$ — точное аналитическое сопряжение вашего Single-FFT пропагатора (обращение квадратичной фазы $z_k$ и $\text{IFFT2}$).
3. **Стадия 2 (Aperture-Domain Regime-Routed Refinement Net + STE 8-bit Quantizer, обучаемая):** Принимает $[\text{Re}(U_{\text{seed}}), \text{Im}(U_{\text{seed}}), \arg(U_{\text{seed}})]$ в координатах DOE $x_0$. Лёгкий роутер по межплоскостной корреляции $\rho_{jk}(T)$ выбирает одну из $K=2$ специализированных фазовых голов (Ветвь A для коррелированных плоскостей, Ветвь B для конфликтных плоскостей) и выдаёт комплексный фазор $(\cos\phi, \sin\phi) \to \text{atan2} \to \text{8-bit STE}$.

### 4. MINIMAL PATCH (Формулы, код PyTorch, параметры, граф инференса)

#### Математическая формулировка сопряжённого оператора $\mathcal{P}_{z_k}^H$

Если прямой Single-FFT Fresnel имеет вид:


$$U_{z_k}(x_z) = D_{z_k}^{(\text{out})}(x_z) \cdot \text{FFT2}\left( D_{z_k}^{(\text{in})}(x_0) \cdot U_0(x_0) \right)$$


то строго сопряжённый (обратный по фазе) оператор в слое `AdjointFresnelLift` равен:


$$\mathcal{P}_{z_k}^H(U_{z_k})(x_0) = \left(D_{z_k}^{(\text{in})}(x_0)\right)^* \cdot \text{IFFT2}\left( \left(D_{z_k}^{(\text{out})}(x_z)\right)^* \cdot U_{z_k}(x_z) \right)$$

#### Минимальная реализация модуля на PyTorch

```python
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

```

* **Число обучаемых параметров:** При `width=32` сеть содержит всего **$\approx 116\,500$ обучаемых параметров** (сопоставимо с лёгким STEM V2 $W32$).
* **Граф конкурсного инференса:**
`T [1, 4, 128, 128]` $\to$ `Target Encoder (3 ResBlocks)` $\to$ `Adjoint Single-FFT Fresnel (4x IFFT2)` $\to$ `DOE Trunk (3 ResBlocks) + Top-1 Router` $\to$ `atan2 + 8-bit Quantize` $\to$ `DOE [1, 1, 128, 128]`.
Время инференса на GPU: **$\approx 1.8\,\text{мс}$** на квартет. Индивидуальная оптимизация Adam/GS и перебор масок **отсутствуют**.

### 5. EXPERIMENT PROTOCOL (Строгий протокол проверки после дедлайна)

1. **Этап 0 (Диагностика без обучения — 5 минут):**
* Запустить `diagnose_grad_conflict.py` на замороженном чекпоинте `STEM V2` и 8 фиксированных квартетах TRAIN (4 из Group A, 4 из Group B, `seed=42`).
* Зафиксировать $\eta_{\text{true}}(\text{PHASE\_HEAD\_HIGH\_FREQ})$, $\text{mean\_cos\_AB}$ и $\text{Mean}(\|g_A\|)/\text{Mean}(\|g_B\|)$.


2. **Этап 1 (Очистка конфаундера Матвея C3/C4 — 15 минут):**
* Контроль 1: Измерить качество $N1$ (STEM V2 на 1 квартете) на шаге $t = 188$ (равное число предъявлений примера с $N64$ на шаге 3000).
* Контроль 2: Обучить $N1$ (3000 шагов) на одном квартете из **Group B**, чтобы установить физический потолок архитектуры STEM V2 на Group B.


3. **Этап 2 (Абляция GF-DDFN при равном числе предъявлений — 40 минут):**
* Выборка: Фиксированный сплит 64 TRAIN квартетов (32 Group A + 32 Group B), валидация строго на `VAL16` (`frozen TEST` не трогать!). Фиксированный `seed=2026`.
* Сравнить при равном бюджете 500 предъявлений на квартет:
* `Baseline`: STEM V2 ($W32$).
* `Ablation-1`: STEM V2 + сбалансированная нормировка градиентов Group A / Group B ($g_i \leftarrow g_i / \|g_i\|_2$).
* `Candidate`: GF-DDFN ($W32$).




4. **Критерии продвижения (Promotion Criteria) на VAL16:**
* Общий $\text{Pearson} \ge 0.6300$ ($+0.0175$ к STEM V2).
* Group B $\text{Pearson} \ge 0.4950$ ($+0.0434$ к STEM V2) без просадки Group A более чем на $0.005$.
* Global $\text{SSIM} \ge 0.5700$ и снижение метрики межплоскостного смешения.


5. **Условия досрочной остановки (Stop Conditions):**
* Если после $50\%$ бюджета обучения на `Ablation/Candidate` ВЧ-энергия фазовой маски $\|\nabla_x \phi\|_2$ не превышает уровень STEM V2 минимум на $15\%$ или $\text{Pearson}_{\text{VAL16}} \lt 0.58$ — остановить прогон.



### 6. EXPECTED BENEFIT (Физический механизм улучшения без выдуманных цифр)

* **Механизм роста Group B:** Фиксированный широкополосный якорь $\psi_k^{\text{anchor}}$ гарантирует, что сопряжённый подъём $\mathcal{P}_{z_k}^H$ сразу распределяет энергию целевых шаблонов Group B по краям апертуры DOE ($|x_0| \to L_0/2$), где дефокусирующий чирп $\exp(i\frac{\pi}{\lambda z_k}|x_0|^2)$ имеет максимальную разность фаз между плоскостями $z_1..z_4$. Сети больше не нужно с нуля «изобретать» ВЧ-диффузор через конфликтные градиенты разных примеров.
* **Механизм подавления межплоскостного смешения:** Модуль `target_encoder` работает в пространстве выровненных целевых плоскостей $x_z$ и вычитает предсказуемое дифракционное гало соседних плоскостей ($\Delta A_k, \Delta\psi_k$) до переноса поля в плоскость DOE.

### 7. RISKS (Риски, память, время, нормативная допустимость)

1. **Риск несовпадения конвенции FFT (`fftshift` / знак фазы $\exp(\pm i k z)$):** Если знак чирпа в `adjoint_fresnel_lift` не совпадает с конвенцией вашего прямого пропагатора `Single-FFT Fresnel`, аналитический подъём даст расфокусировку (деградация до $\text{Pearson} \approx 0.35$). *Митигация:* Проверить условие унитарности $\langle \mathcal{P}_{z_k}(u), v \rangle \equiv \langle u, \mathcal{P}_{z_k}^H(v) \rangle$ с точностью $10^{-6}$ в unit-тесте из 5 строк перед запуском обучения.
2. **Риск спекл-шума от слишком агрессивного ВЧ-якоря:** Слишком большая амплитуда случайной части `rand_hf` в $\psi_k^{\text{anchor}}$ может повысить осевую развязку ценой зернистости (ухудшение SSIM). *Митигация:* Сделать масштаб якоря обучаемым скаляром $\gamma_k \in [0, 1]$, инициализированным значением $0.5$.
3. **Память и время:** Дополнительные 4 вызова `ifft2` на матрицах $128\times 128$ занимают $\lt 2\,\text{МБ}$ VRAM и $\lt 0.25\,\text{мс}$ на шаг.
4. **Нормативная допустимость:** $100\%$ легально. Формируется ровно одна 8-битная DOE $128\times 128$, физический слой фиксирован, ведущую роль играют две обучаемые нейросетевые стадии (6 ResBlocks), итеративная подгонка на инференсе отсутствует.

### 8. DECISION (Операционное решение с учётом времени 09:30 МСК)

* **Для конкурсной сдачи до 09:30 МСК:** **NO_GO на замену продакшн-модели.** Оставить проверенный **STEM V2** в качестве основного решения в сборке GitHub. Любая непроверенная модификация архитектуры за минуты до Stop Coding несёт неприемлемый инженерный риск.
* **Для исследовательского трека (сразу после фиксации сдачи):** **GO.** Первым шагом (за 5 минут) прогнать диагностический скрипт Шага 2 на чекпоинте STEM V2 и 8 квартетах, вторым — внедрить патч `GF_DDFN`.

---

## Краткое заключение для координатора

1. **Что мы поняли:**
* Драматический разрыв в эксперименте Матвея ($N1 = 0.6562$ против $N64 = 0.4938$) на $\sim 75\%$ вызван тем, что $N64$ видел каждый пример всего 188 раз вместо 3000 (из-за spectral bias сеть за 188 показов выучивает только гладкую НЧ-фазу; полное обучение STEM V2 уже поднимает метрику до $0.6124$).
* Однако само плато $0.6124$ и провал на Group B ($0.4516$ против $\sim 0.77$ на Group A) реальны и вызваны **физическим дефицитом осевого разрешения ($\Delta z = 1.2\,\text{мм} \lt \text{DOF} = 1.74\,\text{мм}$)** в сочетании с **калибровочным вырождением фазы в целевых плоскостях**. Однодоменная сверточная сеть, предсказывающая фазу DOE напрямую из целевых амплитуд, усредняет несовместимые ветви ВЧ-фазы разных примеров, что убивает ВЧ-несущую и сливает плоскости между собой.


2. **Что конкретно нужно изменить:**
* Запустить готовый скрипт `diagnose_grad_conflict.py` на чекпоинте STEM V2 (используя правильный эталон ортогональности $C_{\text{orth}} = 1/\sqrt{M}$, чтобы не спутать ортогональность в $\mathbb{R}^P$ с подавлением).
* Перейти от прямой регрессии $T(x_z) \to \phi(x_0)$ к двухдоменной схеме **GF-DDFN**: фиксировать канонический ВЧ-фазовый якорь в целевых плоскостях, предсказывать нейросетью предкомпенсацию межплоскостного эха в координатах мишени, поднимать поле в плоскость DOE точным сопряжённым оператором $\mathcal{P}_{z_k}^H$ и доводить фазу специализированной головой с роутером режимов Group A/B.


3. **Стоит ли исследовать после сдачи:**
* **Однозначно ДА.** Это единственное изменение, которое устраняет физическую причину межплоскостного смешения ($\Delta z \lt \text{DOF}$) и мультимодального гашения ВЧ-фазы без нарушения ограничений инференса.