# Knowledge Distillation for Large Language Models: A Survey of the Prevailing Literature

*Prepared July 2026. Built from a multi-agent deep-research pass (24 sources fetched, 103
claims extracted, 25 top claims adversarially verified 3-0 each; 8 synthesized findings).
Primary sources are cited inline; the verification caveats are listed at the end.*

---

## 1. The classic foundation: logit-based ("word-level") KD

Knowledge distillation as commonly practiced descends from **Hinton, Vinyals & Dean
(2015)** ([arXiv:1503.02531](https://arxiv.org/abs/1503.02531)): train a small *student*
to match the *teacher's* softened output distribution rather than (or in addition to)
the one-hot data labels. The canonical objective mixes hard-label cross-entropy with a
temperature-scaled KL term:

L = α·CE(y, q) + (1−α)·T²·KL(p^(1/T) ‖ q^(1/T))

where p, q are teacher/student next-token distributions, T ≥ 1 flattens both
distributions so that the "dark knowledge" in the teacher's non-argmax probabilities
carries gradient signal, and the T² factor keeps gradient magnitudes comparable across
temperatures. For autoregressive LMs this is applied per token position under teacher
forcing — hence "word-level KD."

Typical practical settings that recur across the literature: T ∈ [1, 4] (T = 2 is the
most common default), α ∈ [0.1, 0.5] on the hard labels, applied over the full
vocabulary at every position.

## 2. Sequence-level KD and the synthetic-data lineage

**Kim & Rush (EMNLP 2016)** ([arXiv:1606.07947](https://arxiv.org/abs/1606.07947))
established the second great branch. Instead of matching per-token distributions, the
teacher *generates* full output sequences (beam search in the original NMT setting) and
the student trains with ordinary cross-entropy on this teacher-generated corpus
(**SeqKD**), optionally interpolated with the original references. Verified results
(each claim confirmed 3-0 against the paper):

- Word-level KD transfers to neural sequence models; SeqKD and sequence-level
  interpolation improve further.
- Their best distilled student ran **~10× faster** than the teacher with only
  **~0.2 BLEU** degradation, beating a from-scratch baseline by **4.2/1.7 BLEU**
  (greedy/beam).
- SeqKD **largely eliminates the need for beam search** at inference — an effect that
  appears even when self-distilling the teacher into itself. Later work on
  non-autoregressive MT (Zhou, Gu & Neubig, ICLR 2020) explains this via *reduced
  multimodality* of the teacher-generated targets.

This lineage leads directly to modern **black-box / synthetic-data distillation** (§5):
when the teacher is an API that exposes only text, SeqKD is essentially the only option.

## 3. Feature, hidden-state, and attention transfer (the BERT era)

A third branch matches *internal* computations rather than outputs. (Note: these
findings were extracted from the primary sources during research but fell below the
adversarial-verification budget cutoff; figures below come from the papers' own
abstracts.)

- **DistilBERT** (Sanh et al., 2019; [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)) —
  triple loss (soft-label logit KD + masked-LM CE + cosine loss aligning hidden states),
  student initialized by taking **every other layer** of the teacher. 40% smaller, 60%
  faster, retains ~97% of GLUE performance.
- **PKD** (Sun et al., 2019; [arXiv:1908.09355](https://arxiv.org/abs/1908.09355)) —
  "patient" distillation from multiple intermediate layers, not just the last.
- **TinyBERT** (Jiao et al., 2020; [arXiv:1909.10351](https://arxiv.org/abs/1909.10351)) —
  layer-wise transfer of embeddings, hidden states, *and attention matrices*, with a
  learned linear projection W to bridge student/teacher widths; two-stage (general +
  task-specific) recipe. A 4-layer student keeps >96.8% of BERT-base GLUE at 7.5×
  smaller / 9.4× faster.
- **MobileBERT** (Sun et al., 2020; [arXiv:2004.02984](https://arxiv.org/abs/2004.02984)) —
  architecture-aware transfer via bottleneck structures; 4.3× smaller / 5.5× faster
  than BERT-base.
- **MiniLM** (Wang et al., 2020; [arXiv:2002.10957](https://arxiv.org/abs/2002.10957)) —
  distills only the **last layer's self-attention** (attention distributions + value
  relations), removing the layer-mapping constraint entirely.

For decoder-only LLMs these techniques are *less* dominant than in the BERT era —
layer-mapping and width-matching get harder with depth, and logit/sequence methods
scale more simply — but hidden-state alignment survives as an auxiliary loss (as
implemented in this repo's `hidden` method).

## 4. The LLM-era divergence debate (2023–2025)

The modern wave re-examined *which divergence* to minimize and *on which data
distribution*. The central verified findings:

### 4.1 Exposure bias motivates on-policy KD

Fixed-dataset, teacher-forced KD trains the student on sequences it never generates
itself, then asks it to condition on its own outputs at inference (**exposure bias /
train–inference distribution mismatch**). This is stated independently by GKD
([arXiv:2306.13649](https://arxiv.org/abs/2306.13649)) and DistiLLM
([arXiv:2402.03898](https://arxiv.org/abs/2402.03898)), corroborated by ImitKD
(Lin et al., EMNLP 2020) — confirmed 6-0 in verification. The fix — training on
student-generated outputs (SGOs) with teacher feedback — raises compute cost sharply,
and that tension structures the 2023–2024 method wave.

### 4.2 MiniLLM: reverse KL

**MiniLLM** (Gu et al., ICLR 2024; [arXiv:2306.08543](https://arxiv.org/abs/2306.08543))
replaces forward KL with **reverse KL** — KL(q‖p) — arguing its *mode-seeking* character
stops the student from placing mass on low-probability teacher regions. Optimization is
policy-gradient-based and on-policy, stabilized by (i) single-step decomposition
(variance reduction), (ii) teacher-mixed sampling (anti reward-hacking), (iii) length
normalization. Students from 120M–13B beat standard KD baselines on
instruction-following with lower exposure bias, better calibration, and retained
diversity. (All claims verified 3-0.)

### 4.3 GKD: generalized JSD + on-policy mixing

**GKD** (Agarwal et al., ICLR 2024; [arXiv:2306.13649](https://arxiv.org/abs/2306.13649))
trains the student on its own self-generated sequences with token-level teacher
feedback, and generalizes the divergence to a β-parameterized **generalized
Jensen–Shannon divergence**, JSD_β(p,q) = β·KL(p‖m) + (1−β)·KL(q‖m) with
m = βp + (1−β)q, spanning forward KL (β→1) to reverse KL (β→0). GKD beats supervised
KD and SeqKD on XSum, WMT, and GSM8K with T5 students; notably, **the best divergence is
task- and student-capacity-dependent**, and the on-policy data distribution often
matters more than the divergence choice. (Verified 9-0 across three claims; mechanism
corroborated by the Hugging Face TRL `GKDTrainer`.)

### 4.4 f-DISTILL: the f-divergence unification

**f-DISTILL** (Wen et al., ACL 2023; [arXiv:2307.15190](https://arxiv.org/abs/2307.15190))
formulates sequence-level KD as minimizing a generalized **f-divergence** —
instantiating KL, reverse KL, JSD, and **total variation distance (TVD)** — shows SeqKD
and ENGINE are approximations of specific instances, and derives a per-token
decomposition making sequence-level divergences tractable. On DART/XSum/WMT16
En-Ro/Commonsense Dialogue, **symmetric divergences (JSD, TVD) beat asymmetric KL/RKL**.
(Verified 12-0 across four claims.)

### 4.5 DistiLLM: skew-KL and efficient off-policy reuse

**DistiLLM** (Ko et al., ICML 2024; [arXiv:2402.03898](https://arxiv.org/abs/2402.03898))
attacks on-policy KD's cost with (i) a **skew-KL** loss, KL(p ‖ αp + (1−α)q), with
provable gradient-stability properties, and (ii) an adaptive off-policy scheme (replay
buffer + validation-loss scheduler) for reusing SGOs. Reports up to **4.3× training
speedup** over MiniLLM/GKD-style on-policy KD with equal or better students on Dolly
Eval / Self-Instruct / Vicuna Eval. (Verified 9-0.)

### 4.6 The corrective: FKL and RKL share an optimum

**Wu et al. (COLING 2025)** ([arXiv:2404.02657](https://arxiv.org/abs/2404.02657))
challenge the mode-seeking narrative: empirically and theoretically, **neither
mode-seeking nor mean-seeking behavior manifests in LLM KD**; FKL and RKL share the
same optimization objective and converge to the same result with enough epochs. What
differs is **early-training dynamics** — FKL fits the *head* of the teacher
distribution first, RKL the *tail* — which matters because practical distillation never
runs to convergence. Their **Adaptive KL (AKL)** weights an FKL/RKL combination by
head/tail gaps with no extra parameters. (Verified 9-0. This *qualifies* the rationale
of MiniLLM/GKD; it does not refute their empirical gains.)

## 5. Black-box / synthetic-data distillation from API teachers

When the teacher exposes only text (no logits), distillation reduces to data
generation + SFT — the dominant route for transferring proprietary-model capability to
open models:

- **Self-Instruct** (Wang et al., ACL 2023;
  [arXiv:2212.10560](https://arxiv.org/abs/2212.10560)) — generate
  instruction/input/output triples from an LM, filter invalid/near-duplicate samples,
  finetune on the rest; seeded by only 175 human-written tasks ("almost
  annotation-free"). Immediately adapted teacher→student by **Alpaca, Vicuna,
  WizardLM, Orca**.
- **Orca** (Mukherjee et al., 2023;
  [arXiv:2306.02707](https://arxiv.org/abs/2306.02707)) — imitates the *reasoning
  process*, not just outputs, using GPT-4 explanation traces — the template for
  **chain-of-thought distillation**, continued at scale by the DeepSeek-R1 distilled
  model family (2025).
- **Distilling step-by-step** (Hsieh et al., 2023;
  [arXiv:2305.02301](https://arxiv.org/abs/2305.02301)) — teacher rationales as
  auxiliary supervision let small task models beat few-shot LLMs with less data.
- **Xu et al. 2024 survey** ([arXiv:2402.13116](https://arxiv.org/abs/2402.13116)) —
  codifies the field into three pillars (algorithms, skill distillation,
  verticalization) and decomposes algorithms into *knowledge elicitation* (labeling,
  expansion, curation, feature, feedback, self-knowledge) + *injection* (SFT,
  divergence/similarity, RL, rank optimization). (Verified 12-0.)

## 6. Practical recipes and evaluation benchmarks

Recurring recipe elements across the verified papers:

| Choice | Common values / practice |
|---|---|
| Temperature T | 1–4 (2 most common); GKD/MiniLLM often use T=1 with on-policy data |
| Hard-label weight α | 0.1–0.5; pure-KD (α=0) common when teacher ≫ student |
| Divergence | FKL default; RKL/JSD/TVD/skew-KL when student is much smaller |
| Student init | From teacher layers when widths match (DistilBERT); random otherwise |
| Data policy | Fixed corpus → cheapest; SGO/on-policy → best but ≥2–4× cost |
| Layer mapping (feature KD) | Uniform spacing + linear projection (TinyBERT) |

Common evaluation suites: perplexity on held-out LM corpora (WikiText-2/103, Pile
subsets); zero-shot loglikelihood tasks via lm-eval-harness (LAMBADA, HellaSwag,
ARC, PIQA, WinoGrande, MMLU); instruction-following judged sets (DollyEval,
Self-Instruct, VicunaEval, AlpacaEval) with ROUGE-L or LLM-as-judge; task suites
(GSM8K, XSum, WMT) for targeted distillation.

## 7. Known failure modes

1. **Exposure bias** (§4.1) — fixed-data KD; addressed by on-policy/SGO training.
2. **Capacity gap** — a much-smaller student cannot represent the teacher's
   distribution; motivates mode-seeking/symmetric divergences (GKD's argument) and
   explains why the best divergence depends on student size. Related: the
   *curse of capacity gap* (Zhang et al.,
   [arXiv:2311.07052](https://arxiv.org/abs/2311.07052)) — a larger teacher is not
   always a better teacher.
3. **Miscalibration** — students overestimate low-probability regions under FKL
   (MiniLLM's argument, qualified by Wu et al.).
4. **Reward hacking** in policy-gradient KD — degenerate outputs the teacher
   scores well; mitigated by teacher-mixed sampling (MiniLLM).
5. **Training collapse with symmetric losses at low budget** — JSD/TVD gradients
   vanish when distributions are far apart (observed in our own experiments; see the
   companion paper).
6. **LLM-as-judge bias** in black-box distillation evaluation — students trained on
   a teacher's outputs then judged by that same teacher inherit systematic bias.

## 8. What remains unsettled (verified open questions)

- How much of on-policy KD's benefit is the *data distribution* vs the *divergence*?
  Does Wu et al.'s same-optimum result extend to on-policy settings?
- Do feature/attention-transfer results (DistilBERT/TinyBERT/MiniLM) replicate at
  modern decoder-only LLM scale?
- Do CoT-trace distillation and logit-level objectives compose?
- How reliable is teacher-as-judge evaluation of students distilled from that teacher?

## Verification caveats

25/25 verified claims were confirmed 3-0 (none refuted). Caveats from the research
pass: feature/hidden-state claims (§3) and concrete hyperparameter recipes were
extracted from primary sources but fell below the adversarial-verification budget;
cross-paper "we beat the predecessor" comparisons are not apples-to-apples (different
teachers/students/datasets/judges); some verifier fetches of arXiv PDFs were
proxy-blocked and used indexed abstracts/mirrors; the field moves quickly (DistiLLM-2,
speculative KD, and R1-era reasoning distillation postdate several findings).

## Source list (fetched primary sources)

Hinton et al. 2015 (1503.02531) · Kim & Rush 2016 (1606.07947) · Sanh et al. 2019
(1910.01108) · Sun et al. 2019 (1908.09355) · Jiao et al. 2020 (1909.10351) · Sun et
al. 2020 (2004.02984) · Wang et al. 2020 (2002.10957) · Gu et al. 2023/24
(2306.08543) · Agarwal et al. 2023/24 (2306.13649) · Wen et al. 2023 (2307.15190) ·
Ko et al. 2024 (2402.03898) · Wu et al. 2024/25 (2404.02657) · Wang et al. 2022/23
(2212.10560) · Mukherjee et al. 2023 (2306.02707) · Hsieh et al. 2023 (2305.02301) ·
Xu et al. 2024 (2402.13116) · Zhang et al. 2023 (2311.07052) · plus surveys
2407.01885, 2410.12896, 2305.12129.
