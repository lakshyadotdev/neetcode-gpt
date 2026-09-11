# 09 — Multi-Layer Backpropagation

## Why this problem exists

Backprop through a single neuron was one short chain: loss → `ŷ` → `z` → `w`. Real
networks are many layers deep, and the gradient has to flow backward through all of
them. This problem is the same idea through a **2-layer MLP**:

```
x → Linear(W1, b1) → ReLU → Linear(W2, b2) → ŷ
```

Two things are new. First, the chain is longer, so gradients pass through a weight
matrix (not just a weight vector) at each step. Second, the activation is **ReLU**,
and its derivative is a **binary mask**: where the pre-activation was positive the
gradient flows through, where it was negative the gradient is killed dead. That mask
is the seed of the Dead ReLU problem.

## Where this gets used

- Any neural network with more than one layer — which is basically all of them.
- `loss.backward()` in PyTorch is this exact process, automated for arbitrarily deep
  networks. Knowing the mask is why you'll later understand dead ReLU detectors and
  why weight initialization (problem 11) matters.

## The setup

Architecture and loss:

```
z1 = x·W1ᵀ + b1          (hidden layer, size hidden_size)
a1 = max(0, z1)          (ReLU)
z2 = a1·W2ᵀ + b2         (output layer, size output_size)
ŷ  = z2                  (no activation on output)
L  = (1/n) Σ (ŷᵢ - yᵢ)²  (MSE)
```

Note each weight matrix is (outputs × inputs): `W1` is (hidden × input), `W2` is
(output × hidden), and the forward pass multiplies `x · Wᵀ`.

Inputs: `x` (1D list), `W1`, `b1`, `W2`, `b2`, `y_true` (1D list). Output: a dict with
`'loss'`, `'dW1'`, `'db1'`, `'dW2'`, `'db2'`, all rounded to **4** decimal places.

### Working the example

`x = [1, 2]`, `W1` = identity matrix, `b1 = [0, 0]`, `W2 = [[0.5, 0.5]]`,
`b2 = [0]`, `y_true = [1]`.

```
forward:  z1 = [1·1 + 2·0, 1·0 + 2·1] = [1, 2]
          a1 = ReLU([1, 2]) = [1, 2]
          z2 = 1·0.5 + 2·0.5 = 1.5
          L  = (1.5 - 1)² = 0.25

backward: dz2 = 2·(1.5 - 1)/1 = 1.0
          dW2 = [1.0] · [1, 2] = [[1.0, 2.0]]
          db2 = [1.0]
          da1 = [1.0] · [0.5, 0.5] = [0.5, 0.5]
          dz1 = da1 ⊙ 1[z1 > 0] = [0.5, 0.5]
          dW1 = [0.5, 0.5]ᵀ · [1, 2] = [[0.5, 1.0], [0.5, 1.0]]
          db1 = [0.5, 0.5]
```

## Main theory

The chain rule, one layer at a time, always using *values you already computed in the
forward pass*. Here's each step and why it has that shape.

### 1. Output gradient: `∂L/∂z2 = 2(z2 - y_true)/n`

`z2` *is* the prediction (no output activation), so this is just the MSE derivative:
the `2` from squaring, the `n` from averaging. With one sample (`n = 1`) it's plain
`2·(prediction - truth)`, i.e. `1.0` in the example.

### 2. Layer-2 weight gradient: `∂L/∂W2 = ∂L/∂z2ᵀ · a1`

A linear layer `z = a·Wᵀ` has `∂z/∂W = a`, so the weight gradient is the **outer
product** of the gradient that arrived at the layer and the layer's input. Shape
check: (outputs,) × (hidden,) → `(outputs × hidden)`, the same shape as `W2` — e.g.
`[1.0]·[1, 2] = [[1, 2]]` in the example.

### 3. Bias gradients: `∂L/∂b = ∂L/∂z`

Because `∂z/∂b = 1`, the bias gradient is just the gradient that arrived at the layer.
No multiplication. `db2 = [1.0]`, and the same recipe gives `db1 = dz1` later.

### 4. Gradient through ReLU: the binary mask

First route the gradient back through the layer: `da1 = dz2 · W2` (the chain rule
through `z2 = a1·W2ᵀ`, i.e. `[1.0]·[0.5, 0.5] = [0.5, 0.5]`). Then comes the new idea:

```
∂z1 = da1 ⊙ 1[z1 > 0]
```

ReLU's derivative is `1` where its input was positive and `0` where it was zero or
negative. The `⊙` means element-wise multiplication. So `∂z1` is `da1` with every
dead neuron's gradient zeroed out. All pre-activations were positive in the example,
so the mask is `[1, 1]` and nothing dies.

### 5 & 6. Layer-1 gradients: same recipes as layer 2

`dW1 = dz1ᵀ · x` (outer product with the network's original input) and `db1 = dz1`.
Same shapes, same reasoning, one layer earlier.

### The dead ReLU warning

If a neuron's pre-activation `z1` is ≤ 0 for *every* input in the training set, its
mask is always 0, its gradient is always 0, and gradient descent can never move it.
It's permanently dead — which is exactly why weight initialization matters (problem
11) and why Leaky ReLU exists.

## The implementation

```
def forward_backward(x, W1, b1, W2, b2, y_true):
    x = np.array(x); W1 = np.array(W1); b1 = np.array(b1)
    W2 = np.array(W2); b2 = np.array(b2); y_true = np.array(y_true)

    # forward pass
    z1 = np.dot(W1, x) + b1
    a1 = np.maximum(0, z1)
    z2 = np.dot(W2, a1) + b2
    loss = np.mean((z2 - y_true) ** 2)

    # backward pass
    n = len(y_true)
    dz2 = 2 * (z2 - y_true) / n
    dW2 = np.outer(dz2, a1)
    db2 = dz2
    da1 = np.dot(dz2, W2)
    mask = (z1 > 0).astype(float)
    dz1 = da1 * mask
    dW1 = np.outer(dz1, x)
    db1 = dz1

    return {'loss': round(float(loss), 4), 'dW1': np.round(dW1, 4),
            'db1': np.round(db1, 4), 'dW2': np.round(dW2, 4),
            'db2': np.round(db2, 4)}
```

Walkthrough of what it does:

1. The `np.array` lines — inputs come in as Python lists; numpy needs arrays.
2. Forward pass — `np.dot(W1, x) + b1` computes `z1` (since `W1` is
   `(hidden × input)`, this matches the spec's `x·W1ᵀ`); `np.maximum(0, z1)` is ReLU;
   then the output layer with no activation.
3. `loss = np.mean((z2 - y_true) ** 2)` — MSE; the `mean` handles the `1/n`.
4. `dz2 = 2 * (z2 - y_true) / n` — output gradient, step 1 of the derivation.
5. `np.outer(dz2, a1)` and `db2 = dz2` — layer-2 gradients: outer product with the
   layer's input, and the arrived gradient for the bias.
6. `da1 = np.dot(dz2, W2)` — route the gradient back through `W2` (step 4's first half).
7. `mask = (z1 > 0).astype(float)` and `dz1 = da1 * mask` — the binary mask:
   `.astype(float)` turns True/False into 1.0/0.0, and `*` zeroes out dead neurons.
8. `np.outer(dz1, x)` and `db1 = dz1` — layer-1 gradients, same outer-product recipe.
9. Round everything to 4 decimals — note this problem uses 4, not 5.

## Where you'll see this again

- **MLP from scratch** (problem 10): this forward pass, minus the backward — computed
  for arbitrary depth.
- **Dead ReLU detector / training** (problems 15-19): the mask here becomes a
  diagnostic, and PyTorch's autograd runs these exact equations on every
  `loss.backward()`.
- **Weight initialization** (problem 11): the mask means ReLU networks need different
  starting weights — the direct follow-up to this problem.

## Gotchas / common mistakes

- **Forgetting the ReLU mask (or applying it to `a1` instead of `z1`)** — treating
  ReLU as identity gives wrong gradients for every neuron that had `z1 ≤ 0`.
- **Wrong outer-product order** — `np.outer(dz, input)`; reversing the arguments
  transposes the matrix and breaks the shape.
- **Forgetting `1/n`** — with one sample it's invisible; with a batch it shrinks
  every gradient by the batch size.
- **Using `>` instead of `>=`** — at `z1 = 0` exactly, ReLU's derivative is
  conventionally 0, so the mask is `z1 > 0`.
- **Rounding to 5 decimals** — this problem rounds to 4.

## One-line summary

Multi-layer backprop is the same chain rule walked backward one layer at a time —
outer products for the weights, the arrived gradient for the biases, and a binary
ReLU mask deciding which neurons' gradients get to live.