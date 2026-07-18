"""Agent E — H9: does KD's advantage over hard-label training persist at 2x budget?

Runs kd and hard for 1200 steps (vs the 600-step baselines in runs/summary.json).
Resumable; invoke repeatedly under `timeout 540 python3 ...` until ALL DONE.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from hypo_common import setup, run_one  # noqa: E402

tok, teacher, train_blocks, eval_blocks = setup()
# ckpt_every=20: at ~8-9 s/step under CPU contention a 540 s invocation cannot
# reach a 50-step boundary after setup, so a finer interval is needed to make
# guaranteed forward progress between invocations.
run_one('kd_1200', teacher, tok, train_blocks, eval_blocks, method='kd', steps=1200, ckpt_every=20)
run_one('hard_1200', teacher, tok, train_blocks, eval_blocks, method='hard', steps=1200, ckpt_every=20)
print('ALL DONE')
