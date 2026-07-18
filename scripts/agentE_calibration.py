"""Agent E — H10: are distilled students better calibrated than the hard-label one?

For each saved student in runs/<method>/student/ computes token-level calibration
on the 200 WikiText-2 eval blocks:
  - ECE: 10 equal-width confidence bins over max next-token probability,
    sum_b (n_b/N) * |acc_b - conf_b|
  - mean confidence (mean max prob) and top-1 next-token accuracy.

Resumable per method; results appended to runs2/calibration.json.
Invoke under `timeout 540 python3 ...` until it prints ALL DONE.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from hypo_common import setup  # noqa: E402

from transformers import GPT2LMHeadModel  # noqa: E402

METHODS = ["hard", "kd", "rkl", "jsd", "tvd", "hidden", "jspace", "seqkd"]
N_BINS = 10
OUT_PATH = os.path.join(ROOT, "runs2", "calibration.json")


@torch.no_grad()
def calibration_stats(model, blocks: torch.Tensor, batch_size: int = 8) -> dict:
    model.eval()
    bin_count = torch.zeros(N_BINS, dtype=torch.float64)
    bin_conf = torch.zeros(N_BINS, dtype=torch.float64)
    bin_correct = torch.zeros(N_BINS, dtype=torch.float64)
    for i in range(0, blocks.size(0), batch_size):
        ids = blocks[i : i + batch_size]
        logits = model(ids).logits[:, :-1].float()
        labels = ids[:, 1:]
        probs = F.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        correct = (pred == labels).double().reshape(-1)
        conf = conf.double().reshape(-1)
        # equal-width bins [0,0.1), ..., [0.9,1.0]
        idx = torch.clamp((conf * N_BINS).long(), max=N_BINS - 1)
        bin_count.index_add_(0, idx, torch.ones_like(conf))
        bin_conf.index_add_(0, idx, conf)
        bin_correct.index_add_(0, idx, correct)
    n = bin_count.sum()
    mask = bin_count > 0
    acc_b = bin_correct[mask] / bin_count[mask]
    conf_b = bin_conf[mask] / bin_count[mask]
    ece = ((bin_count[mask] / n) * (acc_b - conf_b).abs()).sum().item()
    return {
        "ece": ece,
        "mean_confidence": (bin_conf.sum() / n).item(),
        "top1_accuracy": (bin_correct.sum() / n).item(),
        "n_tokens": int(n.item()),
        "bins": [
            {
                "lo": b / N_BINS,
                "hi": (b + 1) / N_BINS,
                "count": int(bin_count[b].item()),
                "conf": (bin_conf[b] / bin_count[b]).item() if bin_count[b] > 0 else None,
                "acc": (bin_correct[b] / bin_count[b]).item() if bin_count[b] > 0 else None,
            }
            for b in range(N_BINS)
        ],
    }


def main():
    tok, teacher, train_blocks, eval_blocks = setup()
    del teacher, train_blocks
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    results = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            results = json.load(f)
    for m in METHODS:
        if m in results:
            print(f"[{m}] already done, skipping", flush=True)
            continue
        path = os.path.join(ROOT, "runs", m, "student")
        model = GPT2LMHeadModel.from_pretrained(path)
        stats = calibration_stats(model, eval_blocks)
        results[m] = stats
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[{m}] ECE={stats['ece']:.4f} conf={stats['mean_confidence']:.4f} "
              f"acc={stats['top1_accuracy']:.4f}", flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    main()
