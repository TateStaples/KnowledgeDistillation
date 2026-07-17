"""Evaluation on local benchmark files (fetched by fetch_data.sh):
  - WikiText-2 test perplexity
  - LAMBADA last-word greedy accuracy
  - HellaSwag / ARC-Easy multiple-choice via length-normalized loglikelihood
"""
from __future__ import annotations

import json
import math
import os

import torch
import torch.nn.functional as F

from .data import DATA_DIR

BENCH_DIR = os.path.join(DATA_DIR, "benchmarks")


@torch.no_grad()
def wikitext_perplexity(model, blocks: torch.Tensor, batch_size: int = 8, device: str = "cpu") -> float:
    model.eval()
    total_nll, total_tok = 0.0, 0
    for i in range(0, blocks.size(0), batch_size):
        ids = blocks[i : i + batch_size].to(device)
        logits = model(ids).logits[:, :-1]
        labels = ids[:, 1:]
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="sum"
        )
        total_nll += nll.item()
        total_tok += labels.numel()
    return math.exp(total_nll / total_tok)


@torch.no_grad()
def _continuation_logprob(model, tokenizer, context: str, continuation: str, device: str) -> tuple[float, int, bool]:
    """Sum log p(continuation | context); returns (logprob, n_tokens, greedy_match)."""
    ctx_ids = tokenizer(context)["input_ids"]
    cont_ids = tokenizer(continuation)["input_ids"]
    ids = torch.tensor([ctx_ids + cont_ids], device=device)
    ids = ids[:, -1024:]
    logits = model(ids).logits[0, :-1]
    logprobs = F.log_softmax(logits.float(), dim=-1)
    n_ctx = ids.size(1) - len(cont_ids)
    lp, greedy = 0.0, True
    for j, tok in enumerate(cont_ids):
        row = logprobs[n_ctx - 1 + j]
        lp += row[tok].item()
        if row.argmax().item() != tok:
            greedy = False
    return lp, len(cont_ids), greedy


def _read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@torch.no_grad()
def lambada_accuracy(model, tokenizer, n_examples: int = 300, device: str = "cpu") -> float:
    """LAMBADA (OpenAI test split): greedy prediction of the final word."""
    ds = _read_jsonl(os.path.join(BENCH_DIR, "lambada_test.jsonl"))
    n = min(n_examples, len(ds))
    correct = 0
    for ex in ds[:n]:
        text = ex["text"]
        ctx, target = text.rsplit(" ", 1)
        _, _, greedy = _continuation_logprob(model, tokenizer, ctx, " " + target, device)
        correct += int(greedy)
    return correct / n


@torch.no_grad()
def multiple_choice_accuracy(
    model, tokenizer, task: str, n_examples: int = 300, device: str = "cpu"
) -> float:
    """Length-normalized loglikelihood selection (lm-eval-harness acc_norm style)."""
    if task == "hellaswag":
        ds = _read_jsonl(os.path.join(BENCH_DIR, "hellaswag_val.jsonl"))

        def get(ex):
            return ex["ctx"], ex["endings"], int(ex["label"])

    elif task == "arc_easy":
        ds = _read_jsonl(os.path.join(BENCH_DIR, "ARC-Easy-Dev.jsonl"))
        ds = [ex for ex in ds if ex["answerKey"] in [c["label"] for c in ex["question"]["choices"]]]

        def get(ex):
            choices = ex["question"]["choices"]
            gold = [c["label"] for c in choices].index(ex["answerKey"])
            ctx = "Question: " + ex["question"]["stem"] + "\nAnswer:"
            return ctx, [c["text"] for c in choices], gold

    else:
        raise ValueError(task)

    correct, n = 0, min(n_examples, len(ds))
    for ex in ds[:n]:
        ctx, choices, gold = get(ex)
        scores = []
        for ch in choices:
            cont = ch if ch.startswith(" ") else " " + ch
            lp, ntok, _ = _continuation_logprob(model, tokenizer, ctx, cont, device)
            scores.append(lp / max(ntok, 1))
        correct += int(max(range(len(scores)), key=lambda i: scores[i]) == gold)
    return correct / n


def evaluate_all(
    model,
    tokenizer,
    eval_blocks: torch.Tensor,
    n_examples: int = 300,
    device: str = "cpu",
) -> dict:
    model.eval()
    return {
        "wikitext2_ppl": wikitext_perplexity(model, eval_blocks, device=device),
        "lambada_acc": lambada_accuracy(model, tokenizer, n_examples, device),
        "hellaswag_acc_norm": multiple_choice_accuracy(model, tokenizer, "hellaswag", n_examples, device),
        "arc_easy_acc_norm": multiple_choice_accuracy(model, tokenizer, "arc_easy", n_examples, device),
    }
