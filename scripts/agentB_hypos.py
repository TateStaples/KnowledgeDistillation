"""Agent B hypotheses.

H3: fixed 50/50 FKL+RKL mixture (AKL-lite, Wu et al. 2024) -> run 'mix_fkl_rkl'
H4: generalized-JSD asymmetry at low budget: beta=0.9 (mass-covering) vs
    beta=0.1 (mode-seeking) -> runs 'jsd_b09', 'jsd_b01'
"""
import sys; sys.path.insert(0, 'scripts')
from hypo_common import setup, run_one

from distill import losses, train


def mixed(s, t, mask, temperature=2.0):
    return 0.5 * losses.forward_kl_loss(s, t, mask, temperature) + \
           0.5 * losses.reverse_kl_loss(s, t, mask, temperature)


tok, teacher, train_blocks, eval_blocks = setup()

# H3: override the 'kd' registry entry ONLY around this call, then restore.
train.LOGIT_LOSSES['kd'] = mixed
try:
    run_one('mix_fkl_rkl', teacher, tok, train_blocks, eval_blocks, method='kd')
finally:
    train.LOGIT_LOSSES['kd'] = losses.forward_kl_loss

# H4: generalized JSD asymmetry.
run_one('jsd_b09', teacher, tok, train_blocks, eval_blocks, method='jsd', jsd_beta=0.9)
run_one('jsd_b01', teacher, tok, train_blocks, eval_blocks, method='jsd', jsd_beta=0.1)

print('ALL DONE')
