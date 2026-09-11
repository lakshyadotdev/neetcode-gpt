# 17 — Training Diagnostics

## Why this problem exists

Your training loop works — the loss goes down. But a model that trains is not a
model that trains *well*: a single bad initialization can silently ruin training.

Real ML debugging is mostly looking at the *insides* of the network: activations
and gradients. Are they exploding? Vanishing? Half the neurons dead? This problem
builds the tools that answer those questions (Karpathy's "Recipe" checklist).

## Where this gets used

- The spec's checklist (loss curve, activations, gradients, dead neurons) is the
  standard debugging procedure in real ML projects.
- TensorBoard / Weights & Biases log these exact numbers while real models train.
- You'll reach for these tools whenever a model won't learn and you don't know why.

## The setup

Three functions that inspect a PyTorch model (a `nn.Sequential` of Linear + ReLU
layers):

1. `compute_activation_stats(model, x)` — forward pass with `torch.no_grad()`;
   per `nn.Linear` layer, record `mean`, `std`, `dead_fraction` of its outputs.
2. `compute_gradient_stats(model, x, y)` — forward + backward using `nn.MSELoss`;
   per `nn.Linear` layer's weight gradient, record `mean`, `std`, `norm`.
3. `diagnose(activation_stats, gradient_stats)` — return one verdict string.

All values rounded to 4 decimal places.

## Main theory

### What activation stats tell you

After a forward pass, each Linear layer produces a bunch of numbers — the
**activations**. Healthy ones have `mean` near 0 and `std` ~1-2. Two failures:

- **Collapsed to 0** (mean and std near 0): information dies before reaching the
  next layer, so the network can't learn.
- **Exploding** (std huge, like 56 in the spec's broken example): numbers blow up
  layer by layer and soon become NaN.

### dead_fraction

A layer's output is a matrix of shape `(batch, neurons)`; a **neuron** is one
column. A neuron is dead if its output is ≤ 0 for *every* sample in the batch:
`(activations <= 0).all(dim=0)` gives a `True`/`False` per neuron, and
`dead_fraction` is the fraction of `True`s. "For all samples" is the key phrase —
a neuron that fires for even one input is still doing something.

### What gradient stats tell you

After `loss.backward()`, every weight has a gradient — the direction and strength
of its update. Each layer's gradient matrix is summarized with:

- `mean` — overall bias in the updates.
- `std` — how spread out the gradients are.
- `norm` — the **L2 norm** (`torch.norm`), the Euclidean length of the flattened
  gradient: "how big are these gradients, overall."

Two failure patterns: **vanishing** (norms near 0 — nothing moves) and
**exploding** (norms huge — updates overshoot and the loss becomes NaN).

### The diagnosis rules

`diagnose` checks, in this exact order:

1. `'dead_neurons'` — any `dead_fraction > 0.5` (half the layer is idle).
2. `'exploding_gradients'` — any gradient `norm > 1000` (updates are enormous).
3. `'vanishing_gradients'` — last layer's gradient `norm < 1e-6` (nothing moves).
4. `'exploding_gradients'` — any activation `std > 10` (catches `N(0, 10)` init).
5. `'healthy'` — none of the above.

Order matters: a network can be exploding *and* full of dead neurons at once.

### What loss curves tell you

The stats tools answer *why*; the loss curve is the *what*:

- **Steadily decreasing** → training is working.
- **Flat the whole time** → learning rate too small, or vanishing gradients.
- **Loss becomes NaN** → exploding gradients (or a data bug).
- **Training low, validation high** → **overfitting** (memorizing, not learning).
- **Both high** → **underfitting** (model or LR too weak for the pattern).

## The implementation

```
import torch
import torch.nn as nn

def compute_activation_stats(model, x):
    stats = []
    with torch.no_grad():
        for module in model:
            x = module(x)
            if isinstance(module, nn.Linear):
                dead = (x <= 0).all(dim=0)
                stats.append({
                    'mean': round(x.mean().item(), 4),
                    'std': round(x.std().item(), 4),
                    'dead_fraction': round(dead.float().mean().item(), 4),
                })
    return stats

def compute_gradient_stats(model, x, y):
    model.zero_grad()
    out = model(x)
    loss = nn.MSELoss()(out, y)
    loss.backward()

    stats = []
    for module in model:
        if isinstance(module, nn.Linear):
            grad = module.weight.grad
            stats.append({
                'mean': round(grad.mean().item(), 4),
                'std': round(grad.std().item(), 4),
                'norm': round(torch.norm(grad).item(), 4),
            })
    return stats

def diagnose(activation_stats, gradient_stats):
    if any(s['dead_fraction'] > 0.5 for s in activation_stats):
        return 'dead_neurons'
    if any(g['norm'] > 1000 for g in gradient_stats):
        return 'exploding_gradients'
    if gradient_stats[-1]['norm'] < 1e-6:
        return 'vanishing_gradients'
    if any(a['std'] > 10.0 for a in activation_stats):
        return 'exploding_gradients'
    return 'healthy'
```

Walkthrough of what it does:

1. `compute_activation_stats`: `torch.no_grad()` stops PyTorch building a graph —
   we only want numbers. Walk the model module by module; when we hit a Linear
   layer, `(x <= 0).all(dim=0)` finds the neurons dead across every sample.
2. `compute_gradient_stats`: `model.zero_grad()` is critical — gradients
   *accumulate*, so stale ones would corrupt the stats. Then forward, MSE loss,
   `backward()` fills every `.grad`. We grab only `module.weight.grad` (the bias
   grad exists too, but the spec asks for the weights).
3. `diagnose`: four `if`s in the spec's order. The vanishing check uses only the
   *last* layer — if even the final layer's gradients are that tiny, nothing is
   learning.

Check the spec's examples: the healthy 3-layer MLP gives activation `std ~1.4`,
gradient norms `~1.9` → no rule fires → `'healthy'`. The broken `N(0, 10)` init
model gives activation `std ~56` → rule 4 fires → `'exploding_gradients'`.

## Where you'll see this again

- **Dead ReLU detector (18)** — zooms in on just the `dead_fraction` metric.
- **Digit classifier (19)** — the tools you run to check your first real net.
- **GPT training (33-34)** — the first things to print when a GPT run goes wrong.

## Gotchas / common mistakes

- **Forgetting `torch.no_grad()`** in the activation pass — works, but wastes
  memory building a graph you never use.
- **Forgetting `model.zero_grad()`** — gradients accumulate; your stats get
  polluted by previous calls.
- **`all` over the wrong dimension** — death is "0 for *all samples*" (`dim=0`).
- **Wrong order in `diagnose`** — the checks have a specific order.
- **Vanishing check on the wrong layer** — the *last* layer's norm, not the first.
- **Rounding inside the computation** — round the final values once, not mid-way.

## One-line summary

Training diagnostics = peek inside the network (activation mean/std, dead
fraction, gradient norms) and let the numbers tell you why it's not learning,
instead of guessing.