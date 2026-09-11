# 15 — RMS Normalization

## Why this problem exists

LayerNorm has two jobs: re-centering (subtract the mean) and re-scaling (divide by the standard deviation). RMS Normalization's insight: the re-centering isn't the important part. What actually helps is the re-scaling. So RMSNorm drops the mean subtraction entirely — and modern LLMs like Llama and Mistral love it because it's simpler, cheaper, and just as effective.

## Where this gets used

- **Llama, Llama 2, Llama 3, Mistral, Gemma** — most modern open-source LLMs use RMSNorm instead of LayerNorm.
- When you see "Pre-RMSNorm" in an architecture diagram, it means RMSNorm applied *before* each attention and feed-forward sublayer (pre-norm style).
- "Pre-norm" (normalize before a sublayer) vs "post-norm" (after) is a design choice. Modern LLMs standardized on pre-norm because it trains more stably — and RMSNorm makes that cheap.
- You'll meet it again in the Build GPT section.

## The setup

NeetCode gives us:

- `x` — a list of floats (feature vector)
- `gamma` — a list of floats (scale parameter, same length as `x`)
- `eps` — default `1e-5`

Notice what's missing: **no `beta`**. LayerNorm has both `gamma` and `beta`; RMSNorm only has `gamma`. That means fewer parameters and less memory.

Output: the normalized list, rounded to 4 decimal places.

Two details worth noticing:

- The output is a plain Python list, not a NumPy array — LayerNorm wanted an array, this one wants a list.
- Rounding is to 4 decimal places, not 5 (that's LayerNorm's precision). These little differences are the kind of thing that trips up the submission.

## Main theory

### Root mean square

"Root mean square" (RMS) is exactly what it sounds like: take the square root of the mean of the squares.

```
RMS(x) = √( (1/n) Σ x_i² + ε )
```

The name is literal — first you *square* each value, then take the *mean* of those squares, then the square *root*. The result is one number summarizing "how big" the vector's values typically are.

The `ε` sits inside the square root, same job as always: it stops a divide-by-zero when every `x_i` is 0.

### The formula

```
x̂_i     = x_i / RMS(x)
output_i = γ_i · x̂_i
```

Compared to LayerNorm:

- No mean subtraction — you never compute `μ`.
- No variance — you only need the mean of squares.
- No `β` — you scale with `γ` and stop.

Note what RMSNorm doesn't do: it never subtracts a mean, so it doesn't center the values at 0. That's fine — for training stability what matters is keeping the *magnitude* under control, and the RMS rescale does exactly that.

One step fewer at every layer, across billions of parameters, adds up to a real speedup — fewer memory reads, less arithmetic, and a computation simple enough to fuse into a single GPU kernel. That's one of the small efficiency wins that let Llama-style models scale at all.

### Why dropping the mean is OK

The LayerNorm authors thought re-centering stabilized training. The RMSNorm paper showed empirically that what actually helps is **scale invariance** — dividing by the RMS so the vector's magnitude stops depending on the input. The mean subtraction costs computation without giving a meaningful benefit, so RMSNorm drops it and stays competitive (sometimes better) with LayerNorm.

Scale invariance concretely: if you multiply every input by 10, the RMS also grows by 10, so `x / RMS(x)` comes out identical. Normalizing `[1, 2, 3]` and normalizing `[10, 20, 30]` produce exactly the same `x̂`. LayerNorm achieves the same property, but it pays for it with the extra mean-and-variance pass — and for transformers, that pass is what RMSNorm skips.

### The cost comparison

|  | LayerNorm | RMSNorm |
|---|---|---|
| Mean computation | Yes | No |
| Variance computation | Yes (needs mean first) | Just mean of squares |
| Learnable parameters | `γ` and `β` (2n) | Only `γ` (n) |
| Used in | Original Transformer, BERT, GPT-2 | Llama, Mistral, Gemma, modern LLMs |

### The worked example

`x = [1.0, 2.0, 3.0]`, `γ = [1, 1, 1]`, `eps = 1e-5`:

- Mean of squares: `(1 + 4 + 9) / 3 = 4.6667`
- RMS: `√(4.6667 + 10⁻⁵) ≈ 2.1602`
- `x̂_1 = 1.0 / 2.1602 ≈ 0.4629`
- `x̂_2 = 2.0 / 2.1602 ≈ 0.9258`
- `x̂_3 = 3.0 / 2.1602 ≈ 1.3887`

Since `γ = [1, 1, 1]`, the output equals `x̂`: `[0.4629, 0.9258, 1.3887]`. If `γ` held other values, each element would be scaled separately — `gamma[0]` scales `x̂_1`, `gamma[1]` scales `x̂_2`, and so on. Notice the output is *not* centered on 0 the way LayerNorm's was — there's no mean subtraction, so RMSNorm doesn't even try to center. It only guarantees the vector's magnitude (the RMS) is about 1.

## The implementation

```
def rms_norm(x, gamma, eps=1e-5):
    x = np.array(x, dtype=float)
    gamma = np.array(gamma, dtype=float)
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return np.round(x / rms * gamma, 4).tolist()
```

Walkthrough:

1. `np.array(..., dtype=float)` — turn the input lists into arrays so the math works element-wise.
2. `x ** 2` — square every element (this is where the "mean of squares" comes from).
3. `np.mean(...)` — average the squares.
4. `np.sqrt(... + eps)` — the RMS, with epsilon inside the root.
5. `x / rms` — the actual normalization: rescale by the RMS.
6. `* gamma` — apply the learnable scale. No beta to add.
7. `np.round(..., 4).tolist()` — round to 4 decimals and return a plain list.

Order matters: `x / rms * gamma` normalizes first, then scales — the same ordering as LayerNorm, just without the shift term.

## Where you'll see this again

- **The Build GPT section** — pre-norm transformers use this style of normalization.
- Every modern LLM architecture you read about (the Llama family especially) will have RMSNorm in the block diagram.
- It's the end of the normalization trio: LayerNorm → BatchNorm → RMSNorm. Same rescaling idea, three different axes/simplifications.

## Gotchas / common mistakes

- **Subtracting the mean** — that's LayerNorm, not RMSNorm. No `μ` anywhere in this formula.
- **Adding a `beta` parameter** — RMSNorm has no shift term. Adding one breaks the signature and doubles the learnable parameters.
- **Forgetting `eps` inside the sqrt** — with all-zero input you'd divide by zero.
- **Order of operations in the RMS** — it's the mean of the *squares*, then square root, not the square of the mean.
- **Rounding to 5 decimals** — this problem wants 4.
- **Applying `gamma` before dividing by RMS** — normalize first, then scale.
- **Rounding before the `gamma` step** — rounding `x / rms` first and then scaling changes the final value. Round once, at the very end.
- **Treating `gamma` as a single scalar** — it's a per-element scale, one weight per feature, so `gamma * x_hat` (or `x / rms * gamma`) uses broadcasting across the whole array.

## One-line summary

RMSNorm rescales a vector by its root mean square — no mean subtraction, no `β`, just a learnable `γ` — which is why modern LLMs like Llama prefer it.