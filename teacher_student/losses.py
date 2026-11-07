"""R16 (#40/#49) teacher->student distillation losses. RESEARCH ONLY, training-time only (never at inference).

Teacher = per-quartet phase-only Adam on TRAIN quartets (classical method used ONLY to make training labels;
organizer: "classical methods for data are allowed"; per-input optimisation at inference is forbidden and not used).

Two candidate distillation terms (one will be chosen in the pre-registration, not both):
  (1) optical_distill: mean_k || x_k(student) - x_k(teacher) ||^2 on the contract display x = |U|/max|U| of each
      plane (native grid). Uses the teacher's REALISABLE reconstructions as targets: invariant to every phase
      ambiguity (global piston, 2*pi wraps, any phase giving the same |U|), so multiple valid phase solutions do not
      conflict.  <- preferred
  (2) circular_phase: mean (1 - cos(phi_s - phi_t - c*)) with the optimal global piston c* = arg(sum e^{i(phi_t-phi_s)})
      removed in closed form. Invariant to 2*pi wraps and global piston only; NOT invariant to other solutions that
      give the same intensities (e.g. conjugate-symmetric/twin-like phases), so it can fight the optical objective.
"""
import torch

DIMS = (-2, -1)


def display(amplitudes: torch.Tensor) -> torch.Tensor:
    a = amplitudes.double()
    return a / a.amax(dim=DIMS, keepdim=True)


def optical_distill(student_amp: torch.Tensor, teacher_amp: torch.Tensor) -> torch.Tensor:
    """mean over batch, planes, pixels of (display(student) - display(teacher))^2; teacher is a constant label."""
    return (display(student_amp) - display(teacher_amp).detach()).pow(2).mean()


def optimal_piston(phi_s: torch.Tensor, phi_t: torch.Tensor) -> torch.Tensor:
    z = torch.exp(1j * (phi_t.double() - phi_s.double())).sum(dim=DIMS, keepdim=True)
    return torch.angle(z)


def circular_phase(phi_s: torch.Tensor, phi_t: torch.Tensor) -> torch.Tensor:
    """mean(1 - cos(phi_s - phi_t + c*)) with c* removed (c* treated as constant w.r.t. phi_s)."""
    c = optimal_piston(phi_s, phi_t).detach()
    return (1 - torch.cos(phi_s.double() - phi_t.double().detach() + c)).mean()
