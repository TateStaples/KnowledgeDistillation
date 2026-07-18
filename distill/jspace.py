"""Jacobian-lens fitting and J-space extraction for the `jspace` distillation target.

Based on Anthropic's Jacobian lens (anthropics/jacobian-lens, Apache-2.0),
companion code to "Verbalizable Representations Form a Global Workspace in
Language Models" (transformer-circuits.pub, 2026):

    lens_l(h) = unembed( J_l @ h ),   J_l = E[ dh_final / dh_l ]

Estimator (reference `jlens.fitting`): for each output dimension, inject a
one-hot cotangent at every valid target position at once and backprop; the
gradient at source position p is sum_{p' >= p} dh_final[p'] / dh_l[p]; take
the mean over source positions and prompts. Early positions are excluded as
attention sinks (SKIP_FIRST_N_POSITIONS = 16), as is the final position.

The J-space (following the jlens-qwen-jspace extension) is the subspace
spanned by the top-k right singular vectors of J_l — the "privileged,
verbalizable" directions of the residual stream at layer l.

Our distillation use: match teacher hidden states only inside this subspace
(see losses.jspace_loss), i.e. emulate the teacher's global-workspace content
rather than its full residual stream.
"""
from __future__ import annotations

import math
import time

import torch

SKIP_FIRST_N_POSITIONS = 16


@torch.enable_grad()
def fit_jacobian_lens(
    teacher,
    blocks: torch.Tensor,
    source_layers: list[int],
    dim_batch: int = 8,
    n_prompts: int = 32,
    device: str = "cpu",
    log_fn=print,
) -> dict[int, torch.Tensor]:
    """Estimate J_l (shape [d_model, d_model], output-dim x source-dim) for each
    source layer. `blocks` are token-id blocks (n, T); the first `n_prompts`
    rows are used as the fitting corpus (the reference implementation notes
    quality saturates quickly; ~100 prompts is usable, paper uses 1000)."""
    teacher.eval()
    d = teacher.config.n_embd
    n_layers = teacher.config.n_layer
    assert all(0 < l < n_layers for l in source_layers)
    jacobians = {l: torch.zeros(d, d) for l in source_layers}
    t0 = time.time()
    for pi in range(n_prompts):
        ids = blocks[pi : pi + 1, :].to(device)
        T = ids.size(1)
        mask = torch.zeros(T, dtype=torch.bool)
        mask[SKIP_FIRST_N_POSITIONS : T - 1] = True
        n_valid = int(mask.sum())

        ids_rep = ids.expand(dim_batch, T)
        out = teacher(ids_rep, output_hidden_states=True)
        hs = out.hidden_states  # [0]=emb, [l]=block l output, [n_layers]=final
        h_final = hs[n_layers]
        sources = [hs[l] for l in source_layers]

        n_chunks = math.ceil(d / dim_batch)
        for ci in range(n_chunks):
            dims = list(range(ci * dim_batch, min((ci + 1) * dim_batch, d)))
            cot = torch.zeros(dim_batch, T, d, device=device)
            for b, dim in enumerate(dims):
                cot[b, mask, dim] = 1.0
            grads = torch.autograd.grad(
                h_final, sources, grad_outputs=cot,
                retain_graph=(ci < n_chunks - 1), allow_unused=False,
            )
            for l, g in zip(source_layers, grads):
                # g[b, p, :] = row of J_l for output dim dims[b], summed over
                # target positions >= p; average over valid source positions
                rows = g[: len(dims)][:, mask, :].mean(dim=1)
                jacobians[l][dims, :] += rows.detach().cpu()
        if (pi + 1) % 4 == 0:
            log_fn(f"[jlens] prompt {pi+1}/{n_prompts} "
                   f"({(time.time()-t0)/(pi+1):.1f}s/prompt)")
    for l in source_layers:
        jacobians[l] /= n_prompts
    return jacobians


def jspace_basis(jacobians: dict[int, torch.Tensor], k: int = 64) -> dict[int, torch.Tensor]:
    """Top-k right singular vectors of each J_l: columns span the J-space.
    Returns {layer: Tensor[d_model, k]}."""
    bases = {}
    for l, J in jacobians.items():
        _, _, Vh = torch.linalg.svd(J.float(), full_matrices=False)
        bases[l] = Vh[:k].T.contiguous()
    return bases


def save_lens(path: str, jacobians: dict[int, torch.Tensor], n_prompts: int) -> None:
    torch.save({"J": {l: J.half() for l, J in jacobians.items()},
                "n_prompts": n_prompts}, path)


def load_lens(path: str) -> dict[int, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return {l: J.float() for l, J in ckpt["J"].items()}
