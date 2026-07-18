import sys; sys.path.insert(0, 'scripts')
import torch
from hypo_common import setup, run_one
from distill.jspace import load_lens, jspace_basis
tok, teacher, train_blocks, eval_blocks = setup()
jac = load_lens('runs/jacobian_lens.pt')
for k in (16, 256):
    run_one(f'jspace_k{k}', teacher, tok, train_blocks, eval_blocks,
            method='jspace', jspace_bases=jspace_basis(jac, k=k), ckpt_every=5)
g = torch.Generator().manual_seed(7)
rand = {l: torch.linalg.qr(torch.randn(768, 64, generator=g)).Q for l in (3, 6, 9)}
run_one('jspace_rand64', teacher, tok, train_blocks, eval_blocks,
        method='jspace', jspace_bases=rand, ckpt_every=5)
print('ALL DONE')
