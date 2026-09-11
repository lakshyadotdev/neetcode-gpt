# 14 — Batch Normalization

## Why this problem exists

LayerNorm normalized across *features within each sample*. Batch Normalization flips the axis: it normalizes across the *batch* for each feature. This was one of the most influential tricks in deep learning (Ioffe & Szegedy, 2015) — it made training much faster and more stable, especially in convolutional and fully-connected networks.

## Where this gets used

- **CNNs and MLPs** — the classic BatchNorm home turf.
- The payoff: you can train with higher learning rates, it's more forgiving of bad weight initialization, and training converges much faster.
- Transformers generally don't use it (they prefer LayerNorm), which is exactly why you need to know both.

## The setup

NeetCode asks for a `BatchNorm1d` from scratch. The inputs:

- `x` — 2-D list, shape `(batch_size, features)`
- `gamma`, `beta` — 1-D lists, length `features`
- `running_mean`, `running_var` — 1-D lists, length `features` (persistent statistics)
- `momentum` — default `0.1`
- `eps` — default `1e-5`
- `training` — `True` during training, `False` at inference

Output: a tuple of three things, all rounded to 4 decimal places — the normalized output, the updated `running_mean`, and the updated `running_var`.

## Main theory

### The axis flip

- **LayerNorm**: normalize across *features* (axis 1) for each sample, on its own.
- **BatchNorm**: normalize across the *batch* (axis 0) for each feature, on its own.

So if your batch has 3 samples with 4 features each, BatchNorm computes 4 separate means and variances — one per feature — each averaged over the 3 samples.

### Training mode

The formulas:

```
μ_B  = (1/N) Σ x_i            # mean across the batch, per feature
σ_B² = (1/N) Σ (x_i - μ_B)²   # variance across the batch
x̂    = (x - μ_B) / √(σ_B² + ε)
y    = γ · x̂ + β
```

Same normalization math as LayerNorm — the only difference is *which direction* you average over.

### Why running statistics exist

During training you use the current batch's mean and variance. But at inference you might process a single sample (batch size 1) — a batch of one has no meaningful statistics. So BatchNorm keeps **running statistics**: an exponential moving average accumulated during training that provides stable estimates for inference. This is the famous "BatchNorm behaves differently in train vs eval" gotcha.

The update rule for each training step:

```
running_mean = (1 - m) · running_mean + m · μ_B
running_var  = (1 - m) · running_var  + m · σ_B²
```

With `m = 0.1`, each new batch contributes 10% of the running estimate. It's a slow, rolling average of what the batch statistics have looked like.

### Inference mode

At inference, swap in the running statistics:

```
x̂ = (x - running_mean) / √(running_var + ε)
```

and still finish with `y = γ·x̂ + β`. You do **not** update the running statistics during inference.

### Edge case: batch size 1

With one sample, the batch variance is exactly 0 — every value equals the mean. The epsilon prevents dividing by zero, and the output collapses to `γ·0 + β = β`. A bit degenerate, but it doesn't crash.

### The worked example

Batch of 3 samples with 4 features:

```
x = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
```

Each column has the same spread, so all four features get the same statistics. Column 0 is `[1, 5, 9]`:

- Mean: `(1 + 5 + 9) / 3 = 5`
- Variance: `((1-5)² + (5-5)² + (9-5)²) / 3 = 32/3 ≈ 10.667`
- Normalized: `(1-5) / √(10.667 + 10⁻⁵) ≈ -1.2247`, `0`, `1.2247`

So the first output row is `[-1.2247, -1.2247, -1.2247, -1.2247]` (sample 1 is the small one in every column), the second row is all zeros, the third is all `1.2247`.

Running statistics update with `m = 0.1`:

```
running_mean = 0.9·[0,0,0,0] + 0.1·[5,6,7,8] = [0.5, 0.6, 0.7, 0.8]
running_var  = 0.9·[1,1,1,1] + 0.1·[10.667, ...] = [1.9667, 1.9667, 1.9667, 1.9667]
```

## The implementation

```
def batchnorm1d(x, gamma, beta, running_mean, running_var, momentum=0.1, eps=1e-5, training=True):
    x = np.array(x, dtype=float)
    gamma = np.array(gamma, dtype=float)
    beta = np.array(beta, dtype=float)
    running_mean = np.array(running_mean, dtype=float)
    running_var = np.array(running_var, dtype=float)

    if training:
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        x_norm = (x - batch_mean) / np.sqrt(batch_var + eps)
        running_mean = (1 - momentum) * running_mean + momentum * batch_mean
        running_var = (1 - momentum) * running_var + momentum * batch_var
    else:
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)

    out = gamma * x_norm + beta
    return (
        np.round(out, 4).tolist(),
        np.round(running_mean, 4).tolist(),
        np.round(running_var, 4).tolist(),
    )
```

Walkthrough:

1. `np.array(..., dtype=float)` — the inputs are plain lists; NumPy turns them into arrays so the math works element-wise.
2. `if training:` — the whole branch structure. Training uses batch statistics and updates the running statistics; inference uses the running statistics and touches nothing.
3. `x.mean(axis=0)` — mean across the batch, per feature. This is the crucial axis: `axis=0` is the batch.
4. `x.var(axis=0)` — variance across the batch. NumPy's default `var` divides by `n`, matching the spec.
5. `x_norm = (x - batch_mean) / np.sqrt(batch_var + eps)` — normalize, with epsilon inside the sqrt.
6. The two running-stat updates — the exponential moving average with `momentum`.
7. `gamma * x_norm + beta` — scale and shift (identical to LayerNorm).
8. `.tolist()` and `np.round(..., 4)` — the spec wants plain lists, rounded to 4 decimals, for all three return values.

## Where you'll see this again

- **RMS Normalization** (problem 15) — the modern simplification that LLMs actually use.
- The LayerNorm vs BatchNorm comparison is worth memorizing: axis, batch dependence, running statistics, train/eval behavior.
- The GPT section mostly uses LayerNorm/RMSNorm — batch statistics are awkward when training on sequences of different lengths.

## Gotchas / common mistakes

- **Wrong axis** — BatchNorm averages over the *batch* (`axis=0`), not the features. Getting this backwards turns it into a weird LayerNorm.
- **Using batch statistics at inference** — a single sample has no meaningful batch stats; use the running ones.
- **Updating running statistics at inference** — they're frozen during eval.
- **Forgetting to return the updated running stats** — the spec wants all three values back.
- **Sample variance (`ddof=1`)** — divides by `n - 1` and the numbers won't match. Use population variance (divide by `n`).
- **Rounding to the wrong precision** — everything here is 4 decimals, not 5 (that's LayerNorm's).
- **Forgetting epsilon** — with batch size 1 you'd divide by zero.

## One-line summary

BatchNorm normalizes each feature across the batch during training and uses running statistics at inference — faster, more stable training at the cost of a training/eval split.