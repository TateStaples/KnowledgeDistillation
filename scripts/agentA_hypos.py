import sys; sys.path.insert(0, 'scripts')
from hypo_common import setup, run_one
tok, teacher, train_blocks, eval_blocks = setup()
run_one('kd_t1', teacher, tok, train_blocks, eval_blocks, method='kd', temperature=1.0, ckpt_every=10)
run_one('kd_t4', teacher, tok, train_blocks, eval_blocks, method='kd', temperature=4.0, ckpt_every=10)
run_one('kd_alpha0', teacher, tok, train_blocks, eval_blocks, method='kd', alpha=0.0, ckpt_every=10)
print('ALL DONE')
