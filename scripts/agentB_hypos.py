"""Agent B hypotheses.

H3: fixed 50/50 FKL+RKL mixture (AKL-lite, Wu et al. 2024) -> run 'mix_fkl_rkl'
H4: generalized-JSD asymmetry at low budget: beta=0.9 (mass-covering) vs
    beta=0.1 (reverse-KL-like / mode-seeking) -> runs 'jsd_b09', 'jsd_b01'

Memory note: 5 agents share 16 GB; the naive full-sequence vocab-sized loss
graph gets this process OOM-killed. All logit losses here are therefore
computed in sequence chunks under torch.utils.checkpoint (recompute-in-
backward). This is mathematically identical to the unchunked loss (verified to
~1e-7): per-chunk masked means are re-weighted by per-chunk mask counts, so
values and gradients match; only peak memory changes.
"""
import sys; sys.path.insert(0, 'scripts')
from hypo_common import setup, run_one

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from distill import losses, train


def mixed(s, t, mask, temperature=2.0):
    """0.5*forward_kl_loss + 0.5*reverse_kl_loss (AKL-lite), fused: each
    log-softmax computed once. Identical to
    0.5*losses.forward_kl_loss(...) + 0.5*losses.reverse_kl_loss(...)."""
    T = temperature
    log_q = F.log_softmax(s / T, dim=-1)
    log_p = F.log_softmax(t / T, dim=-1)
    diff = log_p - log_q
    per_pos = 0.5 * ((log_p.exp() * diff).sum(-1) - (log_q.exp() * diff).sum(-1))
    return losses._masked_mean(per_pos, mask) * (T * T)


def chunked(fn, n_chunks=4):
    """Exact chunked+checkpointed version of a logit loss fn(s, t, mask,
    temperature=..., **kw). Splits along the sequence dim; each chunk's masked
    mean is re-weighted by its mask count so the combined value equals the
    unchunked loss."""
    def wrapped(s, t, mask, temperature=2.0, **kw):
        total_n = mask.sum().clamp(min=1)
        seq = s.size(1)
        step = max(1, (seq + n_chunks - 1) // n_chunks)
        out = 0.0
        for i in range(0, seq, step):
            s_c = s[:, i:i + step]
            t_c = t[:, i:i + step]
            m_c = mask[:, i:i + step]
            if m_c.sum() == 0:
                continue
            l_c = checkpoint(
                lambda a, b, m: fn(a, b, m, temperature=temperature, **kw),
                s_c, t_c, m_c, use_reentrant=False,
            )
            out = out + l_c * m_c.sum()
        return out / total_n
    return wrapped


tok, teacher, train_blocks, eval_blocks = setup()

# H3: override the 'kd' registry entry ONLY around this call, then restore.
train.LOGIT_LOSSES['kd'] = chunked(mixed)
try:
    run_one('mix_fkl_rkl', teacher, tok, train_blocks, eval_blocks, method='kd',
            ckpt_every=25)
finally:
    train.LOGIT_LOSSES['kd'] = losses.forward_kl_loss

# H4: generalized JSD asymmetry. Use the same memory-lean chunked wrapper
# around the shared generalized_jsd_loss (functionally identical), restore
# afterwards.
train.LOGIT_LOSSES['jsd'] = chunked(losses.generalized_jsd_loss)
try:
    run_one('jsd_b09', teacher, tok, train_blocks, eval_blocks, method='jsd',
            jsd_beta=0.9, ckpt_every=25)
    run_one('jsd_b01', teacher, tok, train_blocks, eval_blocks, method='jsd',
            jsd_beta=0.1, ckpt_every=25)
finally:
    train.LOGIT_LOSSES['jsd'] = losses.generalized_jsd_loss

print('ALL DONE')
