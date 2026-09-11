# 18 — Dead ReLU Detector

## Why this problem exists

In the previous problem you computed `dead_fraction` as one metric among many. Now
we zoom in on that single failure mode, because it's sneakier than it sounds: a
neural net can train fine, the loss keeps decreasing, and meanwhile 60% of your
neurons are doing absolutely nothing. The network silently wastes most of its
capacity.

ReLU is the most common activation function, and it has a dangerous failure mode.
This problem builds a dedicated detector for it, plus a rule-based system that
recommends the right fix.

## Where this gets used

- Checking for dead ReLUs is part of the standard training checklist (Karpathy's
  recipe) — you do it whenever training looks stuck or slow.
- Real training frameworks log dead fractions alongside loss curves.
- Modern Transformers (GPT, BERT) use **GELU** instead of ReLU, partly because it
  has no dead zone at all.

## The setup

Two functions:

1. `detect_dead_neurons(model, x)` — forward pass under `torch.no_grad()`. After
   each `nn.ReLU` layer, compute the fraction of neurons whose output is 0 for
   **all** samples in the batch. Return a list of dead fractions (one per ReLU
   layer), each rounded to 4 decimal places.
2. `suggest_fix(dead_fractions)` — given the list of fractions, return one of the
   fix strings.

## Main theory

### Why ReLU neurons die

ReLU is just `max(0, x)`. If a neuron's pre-activation (its weighted sum plus bias)
is negative, it outputs 0. That's fine occasionally — but if it's negative for
*every* input in the batch, the neuron outputs 0 always.

Here's the killer: the gradient of ReLU is **0 for negative inputs**. So a neuron
that always outputs 0 has gradient 0, which means its weights never update. A dead
neuron can **never recover** — its weights stay frozen forever, no matter how long
you train. It's not temporarily off; it's permanently dead.

### Why it happens (and cascades)

- **Learning rate too large** — one big update can push the weights so far that a
  neuron's bias becomes very negative, and it falls into the dead zone.
- **Bad initialization** — the spec's `N(0, 10)` init can kill neurons on the very
  first forward pass.
- **Cascade** — a dead neuron stops feeding its downstream layer, starving those
  neurons of input and making *them* more likely to die too. The damage spreads
  layer by layer.

### Detecting it

A layer's output has shape `(batch, neurons)`. A neuron is a column. Dead means the
output is 0 for every row in that column:

```
dead = (output == 0).all(dim=0)      # one True/False per neuron
dead_fraction = dead.float().mean()  # fraction that are dead
```

A neuron that fires for even one sample is alive — that's why it's `all(dim=0)`,
not `any`.

### The fix decision tree

`suggest_fix` checks in this exact order:

1. **Any layer > 50% dead?** → `'use_leaky_relu'`. Half the layer is useless, so
   switching activation is the real fix. **LeakyReLU** (`f(x) = x` if `x > 0`,
   else `0.01x`) keeps a small non-zero gradient for negative inputs, so a neuron
   can always climb back.
2. **First layer > 30% dead?** → `'reinitialize'`. Death in the first layer is
   catastrophic because it starves everything downstream. Re-initialize with
   **Kaiming init** (`nn.init.kaiming_normal_`), which sizes weights to keep
   activations alive.
3. **Death strictly increases with depth AND last layer > 10%?** →
   `'reduce_learning_rate'`. That pattern says large updates are pushing deeper
   layers into the dead zone, and the effect compounds through layers. Try cutting
   the LR by 10x.
4. **Otherwise** → `'healthy'`.

Other ReLU replacements worth knowing: **PReLU** (like LeakyReLU, but the negative
slope is a learnable parameter), **ELU** (`α(eˣ - 1)` for negative inputs — smooth,
keeps mean activations near 0), and **GELU** (`x·Φ(x)`, smooth, used in GPT/BERT).

## The implementation

```
import torch
import torch.nn as nn

def detect_dead_neurons(model, x):
    dead_fractions = []
    with torch.no_grad():
        for module in model:
            x = module(x)
            if isinstance(module, nn.ReLU):
                dead = (x == 0).all(dim=0)
                dead_fractions.append(round(dead.float().mean().item(), 4))
    return dead_fractions

def suggest_fix(dead_fractions):
    if any(d > 0.5 for d in dead_fractions):
        return 'use_leaky_relu'
    if dead_fractions[0] > 0.3:
        return 'reinitialize'
    if all(dead_fractions[i] < dead_fractions[i + 1]
           for i in range(len(dead_fractions) - 1)) and dead_fractions[-1] > 0.1:
        return 'reduce_learning_rate'
    return 'healthy'
```

Walkthrough of what it does:

1. `detect_dead_neurons`: same pattern as the previous problem — `no_grad`, walk
   the modules, forward through each one. The difference: we record *after* `nn.ReLU`
   layers (not Linear layers), and dead means `== 0` — because after ReLU,
   "output is 0" and "input was ≤ 0" are the same thing. `(x == 0).all(dim=0)`
   gives a boolean per neuron (dead across every batch sample), and
   `.float().mean()` converts it to a fraction.
2. `suggest_fix`: three `if`s in the spec's order.
   - `any(d > 0.5 for d in dead_fractions)` — the severity rule, checked first.
   - `dead_fractions[0] > 0.3` — the early-layer rule. Note it's the *first* ReLU
     layer, not "any" layer.
   - The depth rule is the most specific: `all(prev < next)` for consecutive pairs
     means **strictly** increasing (no ties allowed), *and* the last layer must
     exceed 0.1. Both conditions together → reduce the learning rate.
   - Anything left over → `'healthy'`.

Worked examples:

- `[0.1, 0.2, 0.3]` — strictly increasing, last 0.3 > 0.1 → `'reduce_learning_rate'`.
- `[0.6, 0.4, 0.2]` — 0.6 > 0.5, so the first rule wins → `'use_leaky_relu'`.
- `[0.4, 0.2, 0.1]` — nothing > 0.5, but first layer 0.4 > 0.3 → `'reinitialize'`.
- `[0.05, 0.1, 0.08]` — nothing > 0.5, first not > 0.3, not strictly increasing →
  `'healthy'`.

## Where you'll see this again

- **Training diagnostics (17)** — `dead_fraction` is one of the stats there; this
  problem is the deep dive into that single metric.
- **Digit classifier (19)** — uses ReLU in its hidden layer, exactly the kind of
  network this detector checks.
- **Weight initialization (12)** — Kaiming init exists largely to prevent dead and
  vanishing ReLUs.
- **GPT (final section)** — uses GELU, the transformer-world answer to dead ReLUs.

## Gotchas / common mistakes

- **Checking the wrong layer type** — here we record after `nn.ReLU` layers,
  whereas the previous problem recorded after `nn.Linear` layers.
- **`all` vs `any` over the batch** — a neuron is dead only if its output is 0 for
  *every* sample. Using `any` would flag neurons that are merely quiet sometimes.
- **Wrong dimension** — `all(dim=0)` (across the batch) is the axis that matters.
- **Missing the "strictly" in "strictly increase"** — a list with equal consecutive
  values (e.g. `[0.1, 0.1, 0.2]`) does NOT trigger the depth rule.
- **Reordering the checks** — `suggest_fix` returns the *first* matching rule;
  the order is part of the spec.
- **Thinking a dead neuron can recover** — it can't; its gradient is exactly 0, so
  its weights never move.

## One-line summary

A dead ReLU is a neuron whose input is always negative, so it outputs 0 forever and
never learns — detect it by counting neurons that are 0 for every sample, and fix
it with LeakyReLU, re-initialization, or a lower learning rate.