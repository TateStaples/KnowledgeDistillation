# Agent C: J-space rank profile (H5) and direction specificity (H6)

All runs: 600 steps, batch 8, T=2, alpha=0.5, 4L x 384d student distilled from GPT-2
(124M), WikiText-2 test perplexity. J-space runs reuse the cached Jacobian lens
(`runs/jacobian_lens.pt`, teacher layers {3, 6, 9}); no refitting. New runs live in
`runs2/`, baselines in `runs/summary.json`.

## H5 — "J-space rank has a monotone harm profile: smaller k is less harmful"

**Statement.** Adding a hidden-state matching term hurts this student (kd 179.7 ->
hidden 201.1). If the harm comes from the *amount* of teacher residual stream the
student is forced to emulate, restricting the match to a k-dim J-space subspace
should interpolate monotonically: ppl(k16) <= ppl(k64) <= ppl(k256) <= ppl(hidden/full).

**Setup.** `jspace` method with SVD bases of the cached lens at k = 16 and k = 256
(`jspace_k16`, `jspace_k256`), compared to the k = 64 baseline and the full-rank
analogue `hidden`.

| run | subspace | k | wikitext2 ppl |
|---|---|---|---|
| kd (baseline) | none | 0 | 179.67 |
| jspace_k16 | SVD of J | 16 | TBD |
| jspace (baseline) | SVD of J | 64 | 192.37 |
| jspace_k256 | SVD of J | 256 | TBD |
| hidden (baseline) | full residual | 768 | 201.06 |

**Verdict.** TBD

## H6 — "The J-space DIRECTIONS matter, not just the rank restriction"

**Statement.** jspace (192.4) beats hidden (201.1). Is that because the Jacobian-lens
directions single out the verbalizable-workspace content, or merely because matching
64 dims is a weaker constraint than matching 768? Control: a random orthonormal
64-dim basis per layer (seed 7). If random-64 is about equal to SVD-64, the gap is mere rank
reduction (H6 refuted); if random-64 is clearly worse, the directions carry the value
(H6 supported).

**Setup.** `jspace_rand64`: identical to `jspace` except each layer's basis is the Q
factor of a QR decomposition of a random 768 x 64 Gaussian matrix.

| run | basis | k | wikitext2 ppl |
|---|---|---|---|
| jspace (baseline) | SVD of J (lens) | 64 | 192.37 |
| jspace_rand64 | random orthonormal | 64 | TBD |
| hidden (baseline) | full residual | 768 | 201.06 |

**Verdict.** TBD

## Interpretation w.r.t. the "verbalizable workspace" claim

TBD
