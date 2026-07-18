#!/usr/bin/env python3
"""Run the full distillation comparison: train a student with each objective at
an equal step budget, evaluate on all benchmarks, and write runs/summary.json.

Usage:
    python run_experiments.py --steps 600 --methods hard kd rkl jsd tvd hidden seqkd
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import torch

from distill.data import generate_seqkd_corpus, load_wikitext_blocks
from distill.evaluate import evaluate_all
from distill.models import build_student, count_params, get_tokenizer, load_teacher
from distill.train import train_student

ALL_METHODS = ["hard", "kd", "rkl", "jsd", "tvd", "hidden", "seqkd", "jspace"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS[:-1])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seqkd-sequences", type=int, default=1200)
    ap.add_argument("--jspace-weight", type=float, default=1.0)
    ap.add_argument("--jspace-k", type=int, default=64)
    ap.add_argument("--jspace-layers", type=int, nargs="+", default=[3, 6, 9])
    ap.add_argument("--jspace-prompts", type=int, default=32)
    ap.add_argument("--eval-examples", type=int, default=300)
    ap.add_argument("--max-eval-blocks", type=int, default=200)
    ap.add_argument("--eval-teacher", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(os.cpu_count())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    tok = get_tokenizer()
    teacher = load_teacher(device=device)
    train_blocks = load_wikitext_blocks(tok, "train", args.block_size, max_blocks=4000, seed=args.seed)
    eval_blocks = load_wikitext_blocks(tok, "test", args.block_size, args.max_eval_blocks, args.seed)

    summary_path = os.path.join(args.out, "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    def save():
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    if args.eval_teacher and "teacher" not in summary:
        print("== evaluating teacher ==")
        t0 = time.time()
        metrics = evaluate_all(teacher, tok, eval_blocks, args.eval_examples, device)
        summary["teacher"] = {"params": count_params(teacher), "metrics": metrics,
                              "eval_seconds": time.time() - t0}
        print(json.dumps(metrics, indent=2))
        save()

    seqkd_corpus = None
    for method in args.methods:
        if method in summary:
            print(f"== {method} already done, skipping ==")
            continue
        print(f"== training {method} ==")
        run_dir = os.path.join(args.out, method)
        os.makedirs(run_dir, exist_ok=True)
        torch.manual_seed(args.seed)  # same student init for every method
        student = build_student(teacher=teacher, device=device)

        blocks = train_blocks
        jspace_bases = None
        if method == "jspace":
            from distill.jspace import fit_jacobian_lens, jspace_basis, load_lens, save_lens

            lens_path = os.path.join(args.out, "jacobian_lens.pt")
            if os.path.exists(lens_path):
                jac = load_lens(lens_path)
                print(f"loaded Jacobian lens from {lens_path}")
            else:
                print(f"fitting Jacobian lens at teacher layers {args.jspace_layers}...")
                t0 = time.time()
                jac = fit_jacobian_lens(teacher, train_blocks, args.jspace_layers,
                                        n_prompts=args.jspace_prompts, device=device)
                save_lens(lens_path, jac, args.jspace_prompts)
                print(f"lens fitted in {time.time()-t0:.0f}s -> {lens_path}")
            jspace_bases = jspace_basis(jac, k=args.jspace_k)
        if method == "seqkd":
            corpus_path = os.path.join(args.out, f"seqkd_corpus_{args.seqkd_sequences}.pt")
            if seqkd_corpus is None:
                if os.path.exists(corpus_path):
                    seqkd_corpus = torch.load(corpus_path)
                else:
                    print("generating SeqKD corpus...")
                    t0 = time.time()
                    seqkd_corpus = generate_seqkd_corpus(
                        teacher, tok, train_blocks[: args.seqkd_sequences],
                        block_size=args.block_size, device=device)
                    torch.save(seqkd_corpus, corpus_path)
                    print(f"corpus {tuple(seqkd_corpus.shape)} in {time.time()-t0:.0f}s")
            blocks = seqkd_corpus

        log_lines = []

        def log_fn(msg):
            print(msg, flush=True)
            log_lines.append(msg)

        t0 = time.time()
        history = train_student(
            student, teacher, blocks, method=method, steps=args.steps,
            batch_size=args.batch_size, lr=args.lr, temperature=args.temperature,
            alpha=args.alpha, jspace_weight=args.jspace_weight,
            jspace_bases=jspace_bases, device=device, log_fn=log_fn,
        )
        train_seconds = time.time() - t0

        print(f"== evaluating {method} ==")
        t0 = time.time()
        metrics = evaluate_all(student, tok, eval_blocks, args.eval_examples, device)
        print(json.dumps(metrics, indent=2))

        student.save_pretrained(os.path.join(run_dir, "student"))
        with open(os.path.join(run_dir, "train_log.txt"), "w") as f:
            f.write("\n".join(log_lines))
        summary[method] = {
            "params": count_params(student),
            "steps": args.steps,
            "train_seconds": train_seconds,
            "eval_seconds": time.time() - t0,
            "metrics": metrics,
            "history": history,
            "config": {k: v for k, v in vars(args).items() if k != "methods"},
        }
        save()

    print("all done ->", summary_path)


if __name__ == "__main__":
    main()
