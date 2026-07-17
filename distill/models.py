"""Teacher and student model construction.

The teacher is loaded from a local directory (default models/gpt2, populated by
fetch_data.sh) so no huggingface.co access is required.
"""
from __future__ import annotations

import os

import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TEACHER = os.path.join(REPO_ROOT, "models", "gpt2")


def load_teacher(name: str = DEFAULT_TEACHER, device: str = "cpu") -> GPT2LMHeadModel:
    teacher = GPT2LMHeadModel.from_pretrained(name)
    teacher.eval().to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def build_student(
    n_layer: int = 4,
    n_embd: int = 384,
    n_head: int = 6,
    teacher: GPT2LMHeadModel | None = None,
    init_from_teacher: bool = False,
    device: str = "cpu",
) -> GPT2LMHeadModel:
    """Build a small GPT-2-family student. If `init_from_teacher` and widths match,
    copy embeddings and every k-th teacher block (DistilBERT-style initialization)."""
    cfg = GPT2Config(
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        vocab_size=50257,
        n_positions=1024,
    )
    student = GPT2LMHeadModel(cfg)
    if init_from_teacher and teacher is not None and teacher.config.n_embd == n_embd:
        student.transformer.wte.load_state_dict(teacher.transformer.wte.state_dict())
        student.transformer.wpe.load_state_dict(teacher.transformer.wpe.state_dict())
        stride = teacher.config.n_layer // n_layer
        for i in range(n_layer):
            student.transformer.h[i].load_state_dict(
                teacher.transformer.h[i * stride].state_dict()
            )
        student.transformer.ln_f.load_state_dict(teacher.transformer.ln_f.state_dict())
    student.to(device)
    return student


def get_tokenizer(name: str = DEFAULT_TEACHER) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token
    return tok


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
