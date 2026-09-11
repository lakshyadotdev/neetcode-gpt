# 08 — Backpropagation

## Why this problem exists

Your single neuron can predict, but it can't learn yet. In Gradient Descent you learned
the loop: compute the slope of the loss, take a step downhill. The problem for a neuron
is that the loss doesn't depend on the weights *directly*. The path is:

```
w  →  z (weighted sum)  →  ŷ (activation)  →  L (loss)
```

To nudge a weight we need the slope of `L` with respect to *that weight*, which means
walking the chain backward: `L` depends on `ŷ`, `ŷ` depends on `z`, `z` depends on
`w`. **Backpropagation** is just the **chain rule** applied along this chain, one link
at a time, from the output back to each weight. It's the algorithm that makes all of
deep learning possible.

## Where this gets used

- **Every training run ever** — when PyTorch calls `loss.backward()`, this chain rule
  is what happens under the hood.
- The next problem (multi-layer backprop) is this same idea with a longer chain.
- Understanding backprop is what lets you debug things like slow training and dead
  neurons — you know *why* a gradient is zero.

## The setup

One sigmoid neuron, MSE loss:

```
forward:  z = w·x + b        (weighted sum)
          ŷ = σ(z)           (sigmoid activation)
loss:     L = ½(ŷ - y)²      (mean squared error, half version)
```

We implement `backward()` which returns the gradients:

```
∂L/∂wᵢ = (ŷ - y) · ŷ(1 - ŷ) · xᵢ
∂L/∂b  = (ŷ - y) · ŷ(1 - ŷ)
```

Inputs: `x` (1D array), `w` (1D array), `b` (scalar), `y_true` (scalar). Output: a
tuple `(dL_dw, dL_db)`, all values rounded to 5 decimals.

## Main theory

### The chain rule, in one sentence

To find how the loss changes when a weight changes, **multiply the local slopes along
the path from the loss to that weight**:

```
∂L/∂w = (∂L/∂ŷ) · (∂ŷ/∂z) · (∂z/∂w)
```

Each link is a small, easy derivative. Backprop is the act of computing them one by
one and multiplying them together.

### Link 1 — loss to prediction: `∂L/∂ŷ = ŷ - y`

`L = ½(ŷ - y)²`. The derivative of `½·u²` with respect to `u` is `u`, so
`∂L/∂ŷ = ŷ - y`. That's exactly the error: how far off the prediction was. Notice the
`½` in the loss exists *specifically* so this derivative comes out clean as `ŷ - y`.

### Link 2 — prediction to pre-activation: `∂ŷ/∂z = ŷ(1 - ŷ)`

This is the derivative of the sigmoid. It's a famous identity:

```
σ'(z) = e^(-z) / (1 + e^(-z))²  =  σ(z) · (1 - σ(z))  =  ŷ(1 - ŷ)
```

The neat thing: you can compute the sigmoid's slope from *the output* `ŷ` alone, no
need to go back to `z`. And it tells you something real: when `ŷ` is near 0 or 1, the
sigmoid is flat (slope near 0) — a saturated neuron barely learns.

### Link 3 — pre-activation to weight: `∂z/∂wᵢ = xᵢ`, `∂z/∂b = 1`

`z = w·x + b` is linear, so the slope of `z` with respect to weight `wᵢ` is just the
input `xᵢ` (big inputs → that weight has big influence), and with respect to the bias
it's just `1`.

### Multiplying the chain

```
∂L/∂wᵢ = (ŷ - y) · ŷ(1 - ŷ) · xᵢ
∂L/∂b  = (ŷ - y) · ŷ(1 - ŷ)
```

The shared middle part, `δ = (ŷ - y)·ŷ(1 - ŷ)`, is called the **error signal / delta**:
"how wrong the neuron was, scaled by how awake it is." Every weight gradient is just
`δ · xᵢ` and the bias gradient is `δ` itself.

### Working the example

`x = [1, 2]`, `w = [0.5, 0.5]`, `b = 0`, `y_true = 1`.

```
forward:  z = 1.5,  ŷ = σ(1.5) = 0.81757
error:    ŷ - y = 0.81757 - 1 = -0.18243   (under-predicted: loss shrinks if ŷ grows)
sigmoid:  ŷ(1 - ŷ) = 0.81757 · 0.18243 = 0.14914
delta:    δ = -0.18243 · 0.14914 = -0.02721

∂L/∂w₁ = -0.02721 · 1.0  = -0.02721
∂L/∂w₂ = -0.02721 · 2.0  = -0.05442
∂L/∂b  = -0.02721
```

Both gradients are negative, which makes sense: to push `ŷ` up toward `y = 1` we must
*increase* the weights (the opposite of a negative gradient). And `w₂` gets a bigger
gradient than `w₁` because its input `x₂ = 2` is bigger — it has more leverage on `z`.

## The implementation

```
def backward(x, w, b, y_true):
    z = np.dot(w, x) + b
    y_pred = 1 / (1 + np.exp(-z))
    error = y_pred - y_true
    sigmoid_grad = y_pred * (1 - y_pred)
    delta = error * sigmoid_grad
    dL_dw = delta * x
    dL_db = delta
    return (np.round(dL_dw, 5), round(float(dL_db), 5))
```

Walkthrough of what it does:

1. `z` and `y_pred` — the forward pass, computed once and reused.
2. `error = y_pred - y_true` — link 1 (`∂L/∂ŷ`). Order matters: prediction minus
   truth, not the other way around.
3. `sigmoid_grad = y_pred * (1 - y_pred)` — link 2, computed straight from `ŷ`.
4. `delta = error * sigmoid_grad` — the combined error signal `δ`.
5. `dL_dw = delta * x` — link 3: numpy multiplies element-wise, so each weight gets
   its own `δ·xᵢ`.
6. `dL_db = delta` — the bias gradient is just `δ` (its `∂z/∂b` link is 1).
7. Round everything to 5 decimals and return the tuple `(dL_dw, dL_db)`.

## Where you'll see this again

- **Multi-layer backprop** (problem 09): the exact same chain, but longer, and the
  sigmoid derivative gets replaced by the ReLU mask.
- **Training loop / GPT**: `optimizer.step()` in PyTorch uses these gradients in the
  gradient-descent update you already know: `w = w - lr·∂L/∂w`.
- **Weight initialization** (problem 11): backprop's backward pass multiplies by the
  same weights as the forward pass, which is why bad initialization wrecks gradients.

## Gotchas / common mistakes

- **Swapping the error sign** — it's `ŷ - y`, not `y - ŷ`. The sign tells gradient
  descent which way to step.
- **Forgetting the sigmoid link `ŷ(1 - ŷ)`** — this is the most-missed piece; without
  it you're just doing linear regression's gradient.
- **Forgetting to multiply by `xᵢ`** for the *weight* gradient (the bias gradient does
  not get an `xᵢ`).
- **Treating the `½` as a leftover** — it's already accounted for: the derivative of
  `½(ŷ-y)²` is exactly `ŷ - y`, so no extra `½` appears.
- **Saturated neuron surprise** — when `ŷ` is ~0 or ~1, `ŷ(1-ŷ)` ≈ 0 and learning
  stalls. That's the math, not a bug.

## One-line summary

Backpropagation = the chain rule walked backward from the loss, link by link, telling
each weight exactly how much it contributed to the error so gradient descent can fix
it.