import sys; sys.path.insert(0, 'scripts')
import torch
from hypo_common import setup, run_one
tok, teacher, train_blocks, eval_blocks = setup()
mixed = torch.cat([train_blocks[:1200], torch.load('runs/seqkd_corpus_1200.pt')])
run_one('seqkd_mix50', teacher, tok, mixed, eval_blocks, method='hard')
run_one('kd_deep8x256', teacher, tok, train_blocks, eval_blocks, method='kd',
        student_kwargs=dict(n_layer=8, n_embd=256, n_head=8))
run_one('kd_wide2x512', teacher, tok, train_blocks, eval_blocks, method='kd',
        student_kwargs=dict(n_layer=2, n_embd=512, n_head=8))
print('ALL DONE')
