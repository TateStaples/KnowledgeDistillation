#!/usr/bin/env python3
"""Distill a pretrained causal LM (teacher) into a small student.

Example:
    python distill_model.py --method kd --steps 800 --out runs/kd
    python distill_model.py --method seqkd --steps 800 --out runs/seqkd
    python distill_model.py --method hidden --eval-only runs/hidden

Methods: hard, kd (forward KL), rkl (reverse KL), jsd, tvd, hidden, seqkd, jspace.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from distill.data import generate_seqkd_corpus, load_wikitext_blocks, make_loader  # noqa: F401
from distill.evaluate import evaluate_all
from distill.models import build_student, count_params, get_tokenizer, load_teacher
from distill.train import train_student


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", default="models/gpt2",
                    help="path to local HF-format teacher (see fetch_data.sh)")
    ap.add_argument("--method", default="kd",
                    choices=["hard", "kd", "rkl", "jsd", "tvd", "hidden", "seqkd", "jspace"])
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight on hard-label CE")
    ap.add_argument("--jsd-beta", type=float, default=0.5)
    ap.add_argument("--hidden-weight", type=float, default=1.0)
    ap.add_argument("--jspace-weight", type=float, default=1.0)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=384)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--max-train-blocks", type=int, default=4000)
    ap.add_argument("--max-eval-blocks", type=int, default=200)
    ap.add_argument("--eval-examples", type=int, default=300)
    ap.add_argument("--seqkd-sequences", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output dir (default runs/<method>)")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = args.out or os.path.join("runs", args.method)
    os.makedirs(out_dir, exist_ok=True)

    tok = get_tokenizer(args.teacher)
    teacher = load_teacher(args.teacher, device)
    student = build_student(args.n_layer, args.n_embd, args.n_head, teacher, device=device)
    print(f"teacher params: {count_params(teacher)/1e6:.1f}M  "
          f"student params: {count_params(student)/1e6:.1f}M  device: {device}")

    train_blocks = load_wikitext_blocks(tok, "train", args.block_size,
                                        args.max_train_blocks, args.seed)
    jspace_bases = None
    if args.method == "jspace":
        from distill.jspace import fit_jacobian_lens, jspace_basis, load_lens, save_lens

        lens_path = os.path.join("runs", "jacobian_lens.pt")
        if os.path.exists(lens_path):
            jac = load_lens(lens_path)
        else:
            print("fitting Jacobian lens at teacher layers [3, 6, 9]...")
            os.makedirs("runs", exist_ok=True)
            jac = fit_jacobian_lens(teacher, train_blocks, [3, 6, 9], device=device)
            save_lens(lens_path, jac, 32)
        jspace_bases = jspace_basis(jac, k=64)
    if args.method == "seqkd":
        corpus_path = os.path.join("runs", f"seqkd_corpus_{args.seqkd_sequences}.pt")
        if os.path.exists(corpus_path):
            train_blocks = torch.load(corpus_path)
            print(f"loaded SeqKD corpus {tuple(train_blocks.shape)} from {corpus_path}")
        else:
            print(f"generating SeqKD corpus ({args.seqkd_sequences} sequences)...")
            prompts = train_blocks[: args.seqkd_sequences]
            train_blocks = generate_seqkd_corpus(
                teacher, tok, prompts, prompt_len=16,
                block_size=args.block_size, device=device)
            os.makedirs("runs", exist_ok=True)
            torch.save(train_blocks, corpus_path)
            print(f"saved corpus {tuple(train_blocks.shape)} to {corpus_path}")

    log_path = os.path.join(out_dir, "train_log.txt")
    with open(log_path, "w") as lf:
        def log_fn(msg):
            print(msg)
            lf.write(msg + "\n")
            lf.flush()

        history = train_student(
            student, teacher, train_blocks,
            method=args.method, steps=args.steps, batch_size=args.batch_size,
            lr=args.lr, temperature=args.temperature, alpha=args.alpha,
            jsd_beta=args.jsd_beta, hidden_weight=args.hidden_weight,
            jspace_weight=args.jspace_weight, jspace_bases=jspace_bases,
            device=device, log_fn=log_fn,
        )

    if args.save_model:
        student.save_pretrained(os.path.join(out_dir, "student"))
        tok.save_pretrained(os.path.join(out_dir, "student"))

    results = {"method": args.method, "config": vars(args),
               "student_params": count_params(student), "history": history}
    if not args.skip_eval:
        eval_blocks = load_wikitext_blocks(tok, "test", args.block_size,
                                           args.max_eval_blocks, args.seed)
        print("evaluating...")
        results["metrics"] = evaluate_all(student, tok, eval_blocks,
                                          args.eval_examples, device)
        print(json.dumps(results["metrics"], indent=2))

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_dir}/results.json")


if __name__ == "__main__":
    main()
