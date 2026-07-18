"""Data loading: local WikiText-2 packed into fixed-length blocks for LM training.

Files are fetched by fetch_data.sh (mirrors that avoid huggingface.co).
"""
from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_wikitext_blocks(
    tokenizer,
    split: str = "train",
    block_size: int = 128,
    max_blocks: int | None = None,
    seed: int = 0,
) -> torch.Tensor:
    """Tokenize local WikiText-2, concatenate, and chop into (n_blocks, block_size) ids."""
    path = os.path.join(DATA_DIR, "wikitext-2", f"{split}.txt")
    with open(path) as f:
        text = f.read()
    ids = tokenizer(text, return_tensors=None)["input_ids"]
    n = (len(ids) // block_size) * block_size
    blocks = torch.tensor(ids[:n], dtype=torch.long).view(-1, block_size)
    if max_blocks is not None and blocks.size(0) > max_blocks:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(blocks.size(0), generator=g)[:max_blocks]
        blocks = blocks[idx]
    return blocks


def make_loader(blocks: torch.Tensor, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(TensorDataset(blocks), batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def generate_seqkd_corpus(
    teacher,
    tokenizer,
    prompts: torch.Tensor,
    prompt_len: int = 16,
    block_size: int = 128,
    batch_size: int = 16,
    temperature: float = 1.0,
    device: str = "cpu",
    ckpt_path: str | None = None,
    log_fn=print,
) -> torch.Tensor:
    """Sequence-level KD corpus (Kim & Rush 2016): condition on short prompts from
    the real data and let the teacher complete them; train the student with CE on
    the teacher's outputs. Returns (n, block_size) token ids."""
    out, start = [], 0
    if ckpt_path is not None and os.path.exists(ckpt_path):
        done = torch.load(ckpt_path, weights_only=True)
        out, start = [done], done.size(0)
        log_fn(f"[seqkd] resumed corpus generation at {start}/{prompts.size(0)}")
    for i in range(start, prompts.size(0), batch_size):
        batch = prompts[i : i + batch_size, :prompt_len].to(device)
        gen = teacher.generate(
            batch,
            attention_mask=torch.ones_like(batch),
            max_length=block_size,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
        if gen.size(1) < block_size:  # pad in the rare early-EOS case
            pad = torch.full(
                (gen.size(0), block_size - gen.size(1)), tokenizer.eos_token_id, dtype=torch.long
            )
            gen = torch.cat([gen.cpu(), pad], dim=1)
        out.append(gen[:, :block_size].cpu())
        if ckpt_path is not None:
            torch.save(torch.cat(out, dim=0), ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
    return torch.cat(out, dim=0)
