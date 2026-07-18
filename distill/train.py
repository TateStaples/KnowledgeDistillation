"""Training loop for the distillation objectives."""
from __future__ import annotations

import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from . import losses

LOGIT_LOSSES = {
    "kd": losses.forward_kl_loss,
    "rkl": losses.reverse_kl_loss,
    "jsd": losses.generalized_jsd_loss,
    "tvd": losses.tvd_loss,
}


def _lr_lambda(step, warmup, total):
    if step < warmup:
        return step / max(warmup, 1)
    return max(0.05, (total - step) / max(total - warmup, 1))


def train_student(
    student,
    teacher,
    train_blocks: torch.Tensor,
    method: str = "kd",
    jspace_bases: dict | None = None,
    steps: int = 800,
    batch_size: int = 8,
    lr: float = 3e-4,
    temperature: float = 2.0,
    alpha: float = 0.5,
    jsd_beta: float = 0.5,
    hidden_weight: float = 1.0,
    jspace_weight: float = 1.0,
    device: str = "cpu",
    log_every: int = 50,
    log_fn=print,
):
    """Train `student` with the given objective.

    Methods:
      hard   : CE on data only (no teacher).
      kd     : alpha*CE + (1-alpha)*forward-KL on logits (Hinton).
      rkl    : alpha*CE + (1-alpha)*reverse-KL.
      jsd    : alpha*CE + (1-alpha)*generalized JSD(beta).
      tvd    : alpha*CE + (1-alpha)*total variation distance.
      hidden : kd + hidden_weight * hidden-state MSE (TinyBERT-style).
      seqkd  : CE on teacher-generated corpus (pass that corpus as train_blocks).
      jspace : kd + jspace_weight * J-space emulation loss (match teacher hidden
               states only inside the Jacobian-lens top-k subspace at mapped
               layers; requires jspace_bases from distill.jspace).
    """
    student.train()
    need_teacher = method not in ("hard", "seqkd")
    need_hidden = method in ("hidden", "jspace")

    projections = None
    params = list(student.parameters())
    if need_hidden:
        n_s, n_t = student.config.n_layer, teacher.config.n_layer
        if method == "jspace":
            if not jspace_bases:
                raise ValueError("method 'jspace' requires jspace_bases")
            # map student layers onto the teacher layers the lens was fitted at
            t_layers = sorted(jspace_bases)
            s_layers = [round((i + 1) * n_s / (len(t_layers) + 1)) for i in range(len(t_layers))]
            layer_map = list(zip(s_layers, t_layers))
            jspace_bases = {l: b.to(device) for l, b in jspace_bases.items()}
        else:
            # map student layers 1..n_layer (hidden_states index, 0 = embeddings)
            # to evenly spaced teacher layers, including the last.
            layer_map = [(i + 1, (i + 1) * n_t // n_s) for i in range(n_s)]
        projections = torch.nn.ModuleList(
            [torch.nn.Linear(student.config.n_embd, teacher.config.n_embd) for _ in layer_map]
        ).to(device)
        params += list(projections.parameters())

    opt = AdamW(params, lr=lr, weight_decay=0.01)
    sched = LambdaLR(opt, lambda s: _lr_lambda(s, warmup=steps // 20, total=steps))

    n = train_blocks.size(0)
    g = torch.Generator().manual_seed(1234)
    history, t0 = [], time.time()
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), generator=g)
        ids = train_blocks[idx].to(device)
        inputs, labels = ids[:, :-1], ids[:, 1:]
        mask = torch.ones_like(labels, dtype=torch.float)

        s_out = student(inputs, output_hidden_states=need_hidden)
        s_logits = s_out.logits
        loss_ce = losses.soft_ce_loss(s_logits, labels, mask)

        if need_teacher:
            with torch.no_grad():
                t_out = teacher(inputs, output_hidden_states=need_hidden)
            t_logits = t_out.logits
            if method in LOGIT_LOSSES:
                kwargs = {"temperature": temperature}
                if method == "jsd":
                    kwargs["beta"] = jsd_beta
                loss_kd = LOGIT_LOSSES[method](s_logits, t_logits, mask, **kwargs)
                loss = alpha * loss_ce + (1 - alpha) * loss_kd
            elif method == "hidden":
                loss_kd = losses.forward_kl_loss(s_logits, t_logits, mask, temperature)
                loss_h = losses.hidden_state_loss(
                    s_out.hidden_states, t_out.hidden_states,
                    layer_map, projections, mask,
                )
                loss = alpha * loss_ce + (1 - alpha) * loss_kd + hidden_weight * loss_h
            elif method == "jspace":
                loss_kd = losses.forward_kl_loss(s_logits, t_logits, mask, temperature)
                loss_j = losses.jspace_loss(
                    s_out.hidden_states, t_out.hidden_states,
                    layer_map, projections, jspace_bases, mask,
                )
                loss = alpha * loss_ce + (1 - alpha) * loss_kd + jspace_weight * loss_j
            else:
                raise ValueError(method)
        else:
            loss = loss_ce

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()

        if (step + 1) % log_every == 0 or step == 0:
            history.append({"step": step + 1, "loss": loss.item(), "ce": loss_ce.item()})
            log_fn(
                f"[{method}] step {step+1}/{steps} loss={loss.item():.4f} "
                f"ce={loss_ce.item():.4f} ({(time.time()-t0)/(step+1):.2f}s/step)"
            )
    return history
