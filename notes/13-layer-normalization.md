# 13 — Layer Normalization

## Why this problem exists

When you stack many layers, the numbers flowing through a network can spiral out of control — activations can blow up to huge values or shrink toward zero. Training then becomes unstable and slow. **Layer Normalization** fixes this by re-centering and re-scaling each layer's output so the values stay in a stable range. Every transformer block uses it twice, so you can't build attention — or a GPT — without it.

## Where this gets used

- **Transformers** apply LayerNorm twice per block: after the self-attention sublayer and after the feed-forward sublayer.
- BERT, GPT-2, and the original Transformer all use it.
- Unlike BatchNorm, it works identically in training and inference — no running statistics, and it doesn't care what your batch size is.

## The setup

NeetCode gives us one feature vector and two learnable parameters:

- `x` — a 1-D array of features (the thing to normalize)
- `gamma` — a scale parameter, same length as `x`
- `beta` — a shift parameter, same length as `x`

Output: the normalized vector, rounded to 5 decimal places.

## Main theory

### What "normalizing" actually means

Two numbers describe a list of values:

- **Mean** `μ` — the average. It tells you where the values are centered.
- **Variance** `σ²` — the average squared distance from the mean. It tells you how spread out the values are. Its square root, the **standard deviation** `σ`, is the spread in the original units.

"Normalizing" means reshaping the list so it has mean 0 and standard deviation 1: subtract the mean (that centers it), then divide by the standard deviation (that rescales the spread).

For `x = [1, 2, 3]` the mean is 2, so centering gives `[-1, 0, 1]`. The standard deviation is about `0.8165`, so dividing by it stretches the values to `[-1.22474, 0, 1.22474]` — the exact output of the problem's example.

### The formula

```
x̂_i = (x_i - μ) / √(σ² + ε) · γ_i + β_i
```

with

```
μ  = (1/n) Σ x_i
σ² = (1/n) Σ (x_i - μ)²
```

Each piece, in plain words:

- `x_i - μ` — subtract the mean: centers the values around 0.
- `√(σ² + ε)` — the standard deviation (plus a safety epsilon): the rescaling factor.
- `γ_i` — a learnable **scale**; after normalizing, each feature is re-scaled by its own weight.
- `β_i` — a learnable **shift**; after that, each feature gets its own offset.

### What epsilon is for

If every value in the vector is identical, the variance is 0 and you'd divide by zero. `ε = 10⁻⁵` is a tiny constant inside the square root that keeps the denominator from ever being exactly zero. It's too small to change the answer in any meaningful way, but it prevents a crash (or `nan`) in edge cases.

For example, `x = [5.0, 5.0, 5.0]` has variance 0, so the denominator becomes `√(0 + 10⁻⁵) ≈ 0.00316` — no crash — and every value normalizes to `(5 - 5)/0.00316 = 0`. The normalization step just yields zeros, and `γ`/`β` then scale and shift them as usual.

### Why this helps training

When activations are huge, gradients tend to be huge too (and vice versa), so weight updates overshoot or undershoot unpredictably. Normalized activations keep the numbers in a consistent range, which keeps the gradients sane and training fast and stable. That's the whole reason normalization layers exist.

LayerNorm normalizes each sample *independently* — this is why transformers like it: one sequence's statistics never leak into another, and the math is identical whether your batch has 1 item or 100.

### Why γ and β are learnable

After normalizing, the values have mean 0 and variance 1 — but maybe the network wants different statistics for a particular feature. `γ` and `β` let the model *undo* the normalization if that's useful: if `γ = σ` and `β = μ`, then `γ·x̂ + β` gives back the original `x`. So normalization is a safety net, not a cage.

In practice, `γ` starts at 1 and `β` starts at 0 — the identity transform on already-normalized values — and gradient descent tunes them from there during training. The spec says "learnable" precisely because these aren't constants you pick; the optimizer learns them like any other weight.

### The worked example

`x = [1.0, 2.0, 3.0]`, `γ = [1, 1, 1]`, `β = [0, 0, 0]`:

- Mean: `μ = (1 + 2 + 3) / 3 = 2.0`
- Variance: `σ² = ((1-2)² + (2-2)² + (3-2)²) / 3 = (1 + 0 + 1) / 3 = 0.66667`
- Standard deviation: `√(0.66667 + 10⁻⁵) ≈ 0.81650`

Then each value normalizes to:

- `x̂_1 = (1.0 - 2.0) / 0.81650 = -1.22474`
- `x̂_2 = (2.0 - 2.0) / 0.81650 = 0.0`
- `x̂_3 = (3.0 - 2.0) / 0.81650 = 1.22474`

The full output is `[-1.22474, 0.0, 1.22474]` — nicely centered on 0, spread to unit scale.

Reading the output: `-1.22474` isn't in the original units anymore. It means "this value sits about 1.22 standard deviations below the mean of its feature vector." That unit-agnostic, comparable scale is exactly why downstream layers see stable inputs. (Sanity check: before `γ` and `β` are applied, the normalized vector's mean is 0 and its variance is 1 — that's the definition of "normalized.")

## The implementation

```
def layer_norm(x, gamma, beta, eps=1e-5):
    mu = x.mean()
    var = x.var()
    x_norm = (x - mu) / np.sqrt(var + eps)
    return np.round(x_norm * gamma + beta, 5)
```

Walkthrough:

1. `x.mean()` — `μ`, the mean across the features.
2. `x.var()` — `σ²`, the variance. NumPy's `var()` divides by `n`, which matches the spec's formula exactly (this is "population variance"; some formulas divide by `n - 1` — don't, here).
3. `(x - mu) / np.sqrt(var + eps)` — the actual normalization: subtract the mean, divide by the standard deviation, with epsilon inside the sqrt.
4. `x_norm * gamma + beta` — apply the learnable scale and shift. `gamma` and `beta` are the same length as `x`, so NumPy applies them element by element: `x_norm[0]` is scaled by `gamma[0]` and shifted by `beta[0]`, and so on.
5. `np.round(..., 5)` — NeetCode wants 5 decimal places, and only on the final result.

## Where you'll see this again

- **Batch Normalization** (problem 14) uses the same formula but aggregates over the batch instead of the features.
- **RMS Normalization** (problem 15) is LayerNorm with the mean subtraction removed — and it's what modern LLMs actually use.
- **Every transformer block** in the GPT section normalizes before attention and before the feed-forward network. The block shape is always the same: sublayer → add the residual connection → LayerNorm.

## Gotchas / common mistakes

- **Normalizing over the wrong axis** — LayerNorm goes across the *features of one sample*; over the batch is BatchNorm's job.
- **Using sample variance** — `var(ddof=1)` divides by `n - 1` and would change the numbers. The spec's formula divides by `n`.
- **Forgetting epsilon** — the denominator can be 0 when all inputs are equal, giving a divide-by-zero.
- **Applying γ and β before normalizing** — the order matters: normalize first, then scale and shift.
- **Forgetting γ and β entirely** — the output can't be re-scaled then, and it won't match the spec.
- **Rounding inside the computation** — round only the final output, to 5 decimals.
- **Returning the wrong type** — the spec wants a 1-D NumPy array back (unlike RMSNorm later, which wants a plain list).

## One-line summary

LayerNorm re-centers each sample's features to mean 0 and unit spread, then applies learnable scale `γ` and shift `β` — the stability fix every transformer block uses.