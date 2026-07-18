# The Anthropic "J-Space" Paper: What Exists, What Doesn't, and How We Use It

*Deep-research pass, July 2026: 24 sources, 21 claims adversarially verified (mostly
3-0); key facts independently re-verified by fetching the official repository directly.*

## Verdict: the work is real — but "J-Space" is a concept, not the paper's title

The paper the user is thinking of is:

> **"Verbalizable Representations Form a Global Workspace in Language Models"**
> Anthropic, Transformer Circuits, 2026
> https://transformer-circuits.pub/2026/workspace/index.html
> Announcement: https://www.anthropic.com/research/global-workspace (~July 6, 2026)

There is **no Anthropic paper titled "J-Space."** The J-space is the paper's central
*concept*: a small, privileged zone of internal activity where the model holds concepts
it can report on, reason with, and direct at will — surrounded by a much larger ocean
of automatic processing it cannot access or articulate. It is named after the
**Jacobian lens (J-lens)**, the tool used to discover it.

## The method (verified against the official code)

The **Jacobian lens** reads out what an internal activation is disposed to make the
model say. It linearly transports a residual-stream vector at any layer and position
into the final-layer basis, then decodes it with the model's own unembedding:

```
lens_l(h) = unembed( J_l @ h ),    J_l = E[ ∂h_final / ∂h_l ]
```

The expectation is over prompts, source positions, and all current-and-future target
positions of a generic web-text corpus. The reference estimator injects one-hot
cotangents at every valid target position at once (positions <16 are excluded as
attention sinks, as is the final position) and backprops — one forward pass plus
`ceil(d_model / dim_batch)` backward passes per prompt. The paper fits lenses with
1000 sequences of 128 tokens; quality saturates quickly (~100 prompts usable).

## Open-source implementations (verified via GitHub API + direct fetch)

| Repo | What it is |
|---|---|
| **anthropics/jacobian-lens** | Official companion code. Python, Apache-2.0, created 2026-07-02, ~1.4k stars. Fitting, application, and interactive layer×position visualization. We fetched and read `README.md`, `jlens/fitting.py`, `jlens/lens.py` directly. |
| jerrickhoang/jlens-qwen-jspace | Extends the lens to multimodal Qwen3.5; **defines "J-space" operationally as the subspace spanned by the top-k right singular vectors of J_l** (SVD), with projection-energy and decoding utilities. |
| WeZZard/jlens-qwen36 | MLX / Apple Silicon port (~330 stars). |
| Extraltodeus/J-Wash | Steering tool built on the lens (~170 stars). |
| igorbarshteyn/jlens-gguf | llama.cpp/GGUF port (~70 stars). |
| Festyve/jspace-viz, smartaces/jacobian-lens-for-mac | Visualization / Mac ports. |

Red herring ruled out during research: **DSKD — "Dual-Space Knowledge Distillation"**
(EMNLP 2024, arXiv:2406.17328) is a real distillation paper about aligning
representation *spaces*, but it is not from Anthropic and never uses the term
"J-space" (its closest "J" is Jensen–Shannon divergence support).

## Caveats

- The paper itself (transformer-circuits.pub) could not be fetched directly from this
  sandbox (egress-blocked); its content is known via the official README, the
  announcement-page snapshots, and the companion code — all mutually consistent.
- The J-space-as-top-k-singular-subspace formalization comes from the third-party
  `jlens-qwen-jspace` extension; the official repo ships the lens (J_l matrices)
  and readout, on top of which that definition is a natural construction.
- Six research subagents (including the synthesis stage) hit a session budget limit;
  the surviving claims were individually verified 3-0 (two at 2-1) and the central
  facts (repo existence, method equation, paper title) were re-verified here directly.

## How we use it: "J-space emulation" as a distillation target

The paper is an interpretability work — it defines no distillation objective. Our
adaptation (implemented in `distill/jspace.py` + `losses.jspace_loss`):

1. **Fit the teacher's Jacobian lens** at teacher layers {3, 6, 9} using the reference
   estimator (reimplemented for GPT-2; 32 prompts × 128 tokens, dim_batch 8).
2. **Extract the J-space**: top-k (k=64) right singular vectors of each J_l — the
   directions of layer-l residual space that most influence what the model will say.
3. **Distill inside the J-space**: student hidden states (through a learned 384→768
   projection) are matched to teacher hidden states **only in J-space coordinates**
   (scale-normalized MSE on V_kᵀh), added to the standard forward-KL logit loss.

Hypothesis: compared to full hidden-state matching (`hidden`), the J-space target
concentrates the student's limited capacity on the ~8% of residual directions that
carry the teacher's verbalizable workspace content, discarding the "ocean" of
automatic processing the student cannot afford to replicate anyway. The companion
paper reports the numerical comparison.
