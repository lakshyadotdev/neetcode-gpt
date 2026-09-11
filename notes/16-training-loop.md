# 16 — Training Loop

## Why this problem exists

You've done a lot of building so far: neural nets from scratch in NumPy, PyTorch
basics, and Layer/Batch/RMS norm to keep activations stable. But none of it has
*learned* anything yet. A model that can't learn is just a fancy function with
random numbers inside.

This problem is where it all comes together. The **training loop** is the 5-step
heartbeat of every machine learning system:

1. **Forward pass** — compute predictions.
2. **Loss** — measure how wrong the predictions are.
3. **Backward pass** — compute gradients (how to nudge the weights).
4. **Update** — nudge the weights in the direction that reduces loss.
5. **Repeat** for many epochs.

Everything you've built exists to serve this loop, and everything you'll build next
(the digit classifier, GPT) runs it.

## Where this gets used

- When you later write `optimizer.zero_grad()`, `loss.backward()`,
  `optimizer.step()` in PyTorch, this loop is exactly what's happening under the
  hood.
- The digit classifier (19) and the GPT in the final section both train with this
  loop, scaled up to billions of weights.
- Every real ML project you'll ever see: same 5 steps, repeated over and over.

## The setup

We're training a single-neuron linear regression model using NumPy:

```
ŷ = X · w + b
```

- `X` is a 2D array of shape `(n_samples, n_features)` — the training data.
- `w` is the weight vector (one per feature); `b` is a single bias number.
- `ŷ` is the vector of predicted outputs.

Wrongness is measured with **Mean Squared Error (MSE)**:

```
L = (1/n) Σ (ŷᵢ - yᵢ)²
```

Gradient descent needs the slope of the loss with respect to each parameter. The
spec hands us the two gradients:

```
∂L/∂w = (2/n) · Xᵀ(ŷ - y)
∂L/∂b = (2/n) · Σ(ŷᵢ - yᵢ)
```

And the update rule (same idea as problem 01, just applied to two parameters):

```
w = w - α · ∂L/∂w
b = b - α · ∂L/∂b
```

We start `w` as zeros and `b = 0`, run for `epochs` iterations with learning rate
`lr`, and return the tuple `(w, b)` with all values rounded to 5 decimal places.

## Main theory

### Why MSE

Squaring does two things: every error becomes positive (so big positive and big
negative errors can't cancel out), and big mistakes are punished far harder than
small ones (`10² = 100` vs `1² = 1`). The `1/n` just averages over the dataset.

### Where the gradient formulas come from

Take `∂L/∂w` apart piece by piece:

- `(ŷ - y)` is the vector of errors — how far off each prediction is.
- `Xᵀ` multiplies those errors back onto each feature: since `ŷ = X·w`, feature `j`
  "owns" error proportional to how strongly it appears in the data.
- The `2` comes from differentiating the square (d/dx x² = 2x); the `1/n` averages
  over samples.

`∂L/∂b` is simpler: the bias shifts every prediction equally, so its slope is just
the average error (times 2).

### A concrete first step

Use the spec's example: `X = [[1], [2], [3]]`, `y = [2, 4, 6]`, `epochs = 100`,
`lr = 0.1`. Only one feature, so `w` is a single number.

Step 0: `w = 0`, `b = 0` → predictions are `ŷ = [0, 0, 0]`. Errors:
`ŷ - y = [-2, -4, -6]`.

```
∂L/∂w = (2/3)(1·(-2) + 2·(-4) + 3·(-6)) = (2/3)(-28) ≈ -18.67
∂L/∂b = (2/3)(-2 + -4 + -6) = (2/3)(-12) = -8
```

Updates:

```
w = 0 - 0.1(-18.67) = 1.87
b = 0 - 0.1(-8) = 0.8
```

The data is literally `y = 2x`, so the ideal answer is `w = 2, b = 0`. After 100
epochs with `lr = 0.1` we get `([1.97155], 0.06468)` — close but not exact. Why?
Gradient descent with a fixed learning rate approaches the optimum *asymptotically*:
steps shrink as the slope flattens, so you creep closer forever but never quite
land on it. That's normal, and "good enough" is usually fine.

## The implementation

```
import numpy as np

def train_linear_regression(X, y, epochs, lr):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for _ in range(epochs):
        y_hat = X @ w + b                 # 1. forward pass
        error = y_hat - y                 # 2. errors
        grad_w = (2 / n) * X.T @ error    # 3. gradient for weights
        grad_b = (2 / n) * np.sum(error)  # 3. gradient for bias
        w = w - lr * grad_w               # 4. update
        b = b - lr * grad_b

    return np.round(w, 5), round(b, 5)
```

Walkthrough of what it does:

1. `n, d = X.shape` — we need `n` for the `1/n` in the gradients, and `d` to size
   the weight vector.
2. `w = np.zeros(d)` — the spec requires starting from zeros.
3. `y_hat = X @ w + b` — the forward pass. `X @ w` is a matrix-vector multiply
   giving one prediction per sample; adding `b` shifts every prediction by the bias.
4. `error = y_hat - y` — element-wise difference, the raw error per sample.
5. `grad_w = (2 / n) * X.T @ error` — the matrix version of the formula. `X.T` has
   shape `(features, samples)`, so the multiply produces one slope per weight.
6. `grad_b = (2 / n) * np.sum(error)` — a plain sum (a scalar), because `b` is one
   number.
7. `w = w - lr * grad_w` and `b = b - lr * grad_b` — the exact same
   `x_new = x - lr · slope` from problem 01, applied to each parameter.
8. Round only at the end: the weight vector element-wise, the bias as a scalar.

## Where you'll see this again

- **Handwritten digit classifier (19)** — the same loop with a multi-layer net and
  a classifier loss instead of MSE.
- **Training diagnostics (17)** and **dead ReLU detector (18)** — the tools you
  run *while* this loop runs.
- **GPT training (33-34)** — the same 5 steps with Adam and billions of parameters.

## Gotchas / common mistakes

- **Forgetting the `2`** in `(2/n)` — it comes from differentiating the square, and
  dropping it silently halves every step.
- **Confusing the two gradients** — `∂L/∂w` is a matrix multiply `X.T @ error` (one
  slope per weight); `∂L/∂b` is a plain sum (one slope for the bias).
- **Starting from non-zero weights** — the spec explicitly wants `w` zeros and `b = 0`.
- **Rounding inside the loop** — round only the final result, or the rounding error
  accumulates each epoch.
- **Flipping the error sign** — `y - y_hat` instead of `y_hat - y` reverses every
  update, and the model walks *away* from the optimum.

## One-line summary

Training = repeat 5 steps (predict → measure loss → compute gradients → update
weights → repeat) until the loss stops shrinking — and this loop is all of ML.