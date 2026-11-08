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

