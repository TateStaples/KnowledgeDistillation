"""Distillation loss functions.

All logit-level losses operate on (batch, seq, vocab) tensors and a boolean
mask of valid target positions. Losses are averaged over valid positions.

Conventions:
  p = teacher distribution, q = student distribution.
  Forward KL  = KL(p || q)   (mass-covering, classic Hinton KD)
  Reverse KL  = KL(q || p)   (mode-seeking, MiniLLM-style objective)
  Generalized JSD(beta) interpolates between the two (GKD, Agarwal et al. 2024).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mean(per_pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (per_pos * mask).sum() / mask.sum().clamp(min=1)


def soft_ce_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Standard next-token cross-entropy on hard labels."""
    per_pos = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none"
    ).view(labels.shape)
    return _masked_mean(per_pos, mask)


def forward_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """KL(teacher || student) with temperature scaling, scaled by T^2 (Hinton et al. 2015)."""
    t = temperature
    log_q = F.log_softmax(student_logits / t, dim=-1)
    log_p = F.log_softmax(teacher_logits / t, dim=-1)
    per_pos = (log_p.exp() * (log_p - log_q)).sum(-1)
    return _masked_mean(per_pos, mask) * (t * t)


def reverse_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """KL(student || teacher): mode-seeking objective (teacher-forced MiniLLM variant)."""
    t = temperature
    log_q = F.log_softmax(student_logits / t, dim=-1)
    log_p = F.log_softmax(teacher_logits / t, dim=-1)
    per_pos = (log_q.exp() * (log_q - log_p)).sum(-1)
    return _masked_mean(per_pos, mask) * (t * t)


def generalized_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 2.0,
    beta: float = 0.5,
) -> torch.Tensor:
    """Generalized Jensen-Shannon divergence:
        beta * KL(p || m) + (1-beta) * KL(q || m),  m = beta*p + (1-beta)*q
    beta=0.5 is symmetric JSD; beta->1 approaches forward KL, beta->0 reverse KL.
    """
    t = temperature
    log_q = F.log_softmax(student_logits / t, dim=-1)
    log_p = F.log_softmax(teacher_logits / t, dim=-1)
    p, q = log_p.exp(), log_q.exp()
    log_m = torch.log(beta * p + (1.0 - beta) * q + 1e-10)
    per_pos = beta * (p * (log_p - log_m)).sum(-1) + (1 - beta) * (q * (log_q - log_m)).sum(-1)
    return _masked_mean(per_pos, mask) * (t * t)


def tvd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Total variation distance 0.5*sum|p-q| (Wen et al. 2023, f-divergence KD)."""
    t = temperature
    p = F.softmax(teacher_logits / t, dim=-1)
    q = F.softmax(student_logits / t, dim=-1)
    per_pos = 0.5 * (p - q).abs().sum(-1)
    return _masked_mean(per_pos, mask)


def jspace_loss(
    student_hidden: list[torch.Tensor],
    teacher_hidden: list[torch.Tensor],
    layer_map: list[tuple[int, int]],
    projections: torch.nn.ModuleList,
    bases: dict[int, torch.Tensor],
    mask: torch.Tensor,
) -> torch.Tensor:
    """J-space emulation loss: like hidden_state_loss, but the mismatch is
    measured only inside the teacher's J-space at each mapped teacher layer —
    the top-k right-singular subspace of the Jacobian lens J_l (the
    "verbalizable workspace" directions), rather than the full residual stream.

    `bases[t_idx]` is the [d_teacher, k] orthonormal J-space basis for teacher
    layer t_idx (see distill.jspace).
    """
    total = 0.0
    m = mask.unsqueeze(-1)
    for i, (s_idx, t_idx) in enumerate(layer_map):
        V = bases[t_idx]  # [d_t, k]
        s = projections[i](student_hidden[s_idx])
        tgt = teacher_hidden[t_idx]
        s_c, t_c = s @ V, tgt @ V  # J-space coordinates [B, T, k]
        scale = (t_c**2).mean().detach().clamp(min=1e-6)
        per = ((s_c - t_c) ** 2).mean(-1, keepdim=True) / scale
        total = total + (per * m).sum() / m.sum().clamp(min=1)
    return total / max(len(layer_map), 1)


def hidden_state_loss(
    student_hidden: list[torch.Tensor],
    teacher_hidden: list[torch.Tensor],
    layer_map: list[tuple[int, int]],
    projections: torch.nn.ModuleList,
    mask: torch.Tensor,
) -> torch.Tensor:
    """TinyBERT-style hidden-state matching: MSE between projected student states
    and teacher states at mapped layers. `projections[i]` maps student width to
    teacher width for the i-th mapped pair."""
    total = 0.0
    m = mask.unsqueeze(-1)
    for i, (s_idx, t_idx) in enumerate(layer_map):
        s = projections[i](student_hidden[s_idx])
        tgt = teacher_hidden[t_idx]
        # normalize by the teacher's mean squared activation so the loss is
        # scale-invariant and comparable in magnitude to the logit losses
        scale = (tgt**2).mean().detach().clamp(min=1e-6)
        per = ((s - tgt) ** 2).mean(-1, keepdim=True) / scale
        total = total + (per * m).sum() / m.sum().clamp(min=1)
    return total / max(len(layer_map), 1)
