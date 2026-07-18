"""Shared driver for the hypothesis-testing subagents.

Each hypothesis run is fully resumable: training checkpoints every 50 steps
(distill.train), the final student is saved before evaluation, and completed
runs are skipped. Invoke scripts repeatedly under `timeout 540 ...` until they
print ALL DONE.

Results: runs2/<name>.json  (perplexity always; benchmark tasks optional).
Baselines for comparison are in runs/summary.json.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

torch.set_num_threads(1)  # 5 agents share 4 cores

from distill.data import load_wikitext_blocks  # noqa: E402
from distill.evaluate import evaluate_all, wikitext_perplexity  # noqa: E402
from distill.models import build_student, count_params, get_tokenizer, load_teacher  # noqa: E402
from distill.train import train_student  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs2")


def setup(block_size: int = 128, max_train_blocks: int = 4000, seed: int = 0):
    torch.manual_seed(seed)
    tok = get_tokenizer()
    teacher = load_teacher()
    train_blocks = load_wikitext_blocks(tok, "train", block_size, max_train_blocks, seed)
    eval_blocks = load_wikitext_blocks(tok, "test", block_size, 200, seed)
    return tok, teacher, train_blocks, eval_blocks


def run_one(
    name: str,
    teacher,
    tok,
    train_blocks,
    eval_blocks,
    method: str = "kd",
    steps: int = 600,
    student_kwargs: dict | None = None,
    full_benchmarks: bool = False,
    seed: int = 0,
    **train_kwargs,
) -> dict | None:
    """Train + evaluate one configuration; returns the result dict (or the
    cached one if already complete). Safe to kill and re-invoke at any point."""
    os.makedirs(OUT, exist_ok=True)
    res_path = os.path.join(OUT, f"{name}.json")
    if os.path.exists(res_path):
        with open(res_path) as f:
            return json.load(f)

    torch.manual_seed(seed)
    student = build_student(**(student_kwargs or {}), teacher=teacher)
    final_path = os.path.join(OUT, f"{name}_final.pt")
    if os.path.exists(final_path):
        student.load_state_dict(torch.load(final_path, weights_only=True))
        history = []
        print(f"[{name}] training already complete, evaluating")
    else:
        history = train_student(
            student, teacher, train_blocks, method=method, steps=steps,
            ckpt_path=os.path.join(OUT, f"{name}_ckpt.pt"),
            log_fn=lambda m: print(f"[{name}] {m}", flush=True),
            **train_kwargs,
        )
        torch.save(student.state_dict(), final_path)

    if full_benchmarks:
        metrics = evaluate_all(student, tok, eval_blocks, n_examples=150)
    else:
        metrics = {"wikitext2_ppl": wikitext_perplexity(student, eval_blocks)}
    result = {
        "name": name, "method": method, "steps": steps,
        "params": count_params(student),
        "student_kwargs": student_kwargs or {},
        "train_kwargs": {k: str(v)[:80] for k, v in train_kwargs.items()},
        "metrics": metrics, "history": history,
    }
    with open(res_path, "w") as f:
        json.dump(result, f, indent=2)
    for p in (os.path.join(OUT, f"{name}_ckpt.pt"), final_path):
        if os.path.exists(p):
            os.remove(p)
    print(f"[{name}] DONE {metrics}")
    return result


def baselines() -> dict:
    with open(os.path.join(os.path.dirname(OUT), "runs", "summary.json")) as f:
        return json.load(f)
