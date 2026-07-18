# Knowledge Distillation for LLMs

A study of knowledge-distillation methods for large language models: a literature
report, a working implementation of the most common KD objectives, benchmark
results, and a LaTeX write-up.

## Contents

- `report/distillation_literature_report.md` — survey of the prevailing KD literature
- `report/jspace_research.md` — investigation of the "J-Space" paper question
- `distill/` — library: losses, models, data, training loop, evaluation
- `distill_model.py` — CLI to distill the pretrained teacher with one method
- `run_experiments.py` — full comparison across all methods + benchmarks
- `paper/paper.tex` — LaTeX paper with methodology and numerical results
- `fetch_data.sh` — downloads the GPT-2 teacher + all benchmark data from
  mirrors that do not require huggingface.co

## Setup

```bash
pip install -r requirements.txt
bash fetch_data.sh          # teacher weights + WikiText-2 + LAMBADA/HellaSwag/ARC
```

## Distill a pretrained model

```bash
python distill_model.py --method kd --steps 600         # Hinton forward-KL KD
python distill_model.py --method rkl                    # reverse KL (MiniLLM-style)
python distill_model.py --method jsd --jsd-beta 0.5     # generalized JSD (GKD-style)
python distill_model.py --method hidden                 # + hidden-state matching
python distill_model.py --method seqkd                  # sequence-level KD
python distill_model.py --method jspace                 # + J-space auxiliary target
```

Teacher: GPT-2 124M. Student: 4-layer/384-dim GPT-2 (~29M params).

## Reproduce the paper's numbers

```bash
python run_experiments.py --steps 600 --eval-teacher \
    --methods hard kd rkl jsd tvd hidden seqkd jspace
```

Results land in `runs/summary.json`; each method's training log and checkpoint in
`runs/<method>/`.

## Results (600 steps, single seed, CPU)

| method | params | WT-2 ppl ↓ | LAMBADA | HellaSwag | ARC-E |
|---|---|---|---|---|---|
| teacher (GPT-2) | 124.4M | 53.5 | .343 | .317 | .333 |
| hard | 26.8M | 191.2 | .000 | .227 | .263 |
| **kd** (forward KL) | 26.8M | **179.7** | .000 | .227 | .257 |
| rkl | 26.8M | 194.3 | .000 | .230 | .253 |
| jsd (β=0.5) | 26.8M | 182.1 | .000 | .230 | .250 |
| tvd | 26.8M | 186.3 | .000 | .220 | .243 |
| hidden | 26.8M | 201.1 | .000 | .230 | .243 |
| seqkd | 26.8M | 390.4 | .000 | .227 | .233 |
| jspace (k=64) | 26.8M | 192.4 | .000 | **.237** | .253 |

Takeaways: forward-KL logit KD wins in the low-budget regime; symmetric
divergences (JSD/TVD) are close; reverse KL loses to the no-teacher baseline
(tail-first early dynamics, cf. Wu et al. 2024); pure SeqKD collapses on
real-data perplexity; and the J-space auxiliary target (matching teacher hidden
states only in the Jacobian-lens top-64 subspace, after Anthropic's 2026
global-workspace paper) is markedly less harmful than full hidden-state
matching — see `paper/paper.pdf` for the full analysis.

## Tests

```bash
python -m pytest tests/
```
