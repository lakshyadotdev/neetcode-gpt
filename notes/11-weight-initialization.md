# 11 — Weight Initialization

## Why this problem exists

You can build an MLP now. But try it: stack 10 layers of random N(0,1) weights and
watch the activations — they either explode toward infinity or collapse to zero. After
10 layers the signal is astronomically large or numerically dead, and that's one of
the most common reasons deep networks fail to train.

The fix is deceptively simple: don't pick the starting weights arbitrarily — pick them
so that each layer *preserves the variance* of what flows through it. That technique,
**weight initialization**, is one of the things that made deep learning practical.

## Where this gets used

- **Every training run** — weights have to start *somewhere*, and "somewhere" decides
  whether the network trains at all.
- PyTorch uses **Kaiming initialization by default** for `nn.Linear` layers (see the
  `torch.nn.init` module for doing it by hand) — and understanding it explains the
  Dead ReLU problem from the last note: bad starts kill neurons before training begins.

## The setup

Three functions to implement:

1. `xavier_init(fan_in, fan_out)` — return a `(fan_out × fan_in)` weight matrix using
   **Xavier (Glorot)** normal initialization.
2. `kaiming_init(fan_in, fan_out)` — return a `(fan_out × fan_in)` weight matrix using
   **Kaiming (He)** normal initialization.
3. `check_activations(num_layers, input_dim, hidden_dim, init_type)` — forward a random
   input through `linear + ReLU` per layer and return the **standard deviation of the
   activations at each layer**; `init_type` is `'xavier'`, `'kaiming'`, or `'random'`.

For reproducibility, call `torch.manual_seed(0)` at the start of each function. Two
names you'll need: `fan_in` = number of inputs a layer receives, `fan_out` = number of
outputs it produces. For a `(fan_out × fan_in)` matrix, `fan_in` is the number of
*columns* and `fan_out` the number of *rows*.

### Working the example

`xavier_init(fan_in=4, fan_out=3)`:

```
std = √(2 / (4 + 3)) = √(2/7) = 0.5345
matrix = torch.randn(3, 4) * 0.5345   (with torch.manual_seed(0))
```

With the seed fixed, that gives the `3 × 4` matrix:

```
[[ 0.8237, -0.1568, -1.1646,  0.3038],
 [-0.5797, -0.7476,  0.2156,  0.4479],
 [-0.3845, -0.2156, -0.3189,  0.0973]]
```

Each value is drawn from a normal with mean 0 and std 0.5345.

## Main theory

### Why random weights break

A layer computes `h_out = h_in · W`. If the weights are too *large*, the outputs grow
as they pass through each layer — multiply and multiply and after 10 layers you have
gigantic numbers. If too *small*, everything shrinks toward zero and the signal dies.
Backprop has the same problem in *reverse*: the backward pass multiplies by the same
weight matrices, so bad weights also kill gradients before they reach the first layers.

### The fix: preserve variance

**Xavier initialization** (Glorot & Bengio, 2010, for sigmoid/tanh):

```
std = √(2 / (fan_in + fan_out))
```

The denominator *averages* the input and output dimensions. Why both? The forward pass
multiplies by `fan_in` values, the backward pass by `fan_out` — averaging keeps the
signal stable in both directions. Result: variance in ≈ variance out, however deep.

### Why Kaiming doubles the numerator (the whole point of the 2)

**Kaiming initialization** (He et al., 2015, for ReLU networks):

```
std = √(2 / fan_in)
```

ReLU sets every negative activation to zero, which *cuts the variance roughly in half*
at every layer — the negative half of the distribution just disappears. Kaiming
compensates with the extra factor of 2 in the numerator. Compared to Xavier it's
`√2 ≈ 1.41×` bigger — exactly the correction for "ReLU killed half the neurons."

### The diagnostic

`check_activations` makes the whole problem visible in one number. Forward a random
input through `num_layers` of `Linear + ReLU` and record the std of the activations
after each layer:

- `'random'` (std 1) — the stds **explode** layer by layer (e.g. ~1.6, ~5, ~26, ~120,
  ... past 500 by layer 10). Training on those numbers is hopeless.
- `'xavier'` — much better, but still drifts down for ReLU networks because of the
  half-killed variance (the "wrong tool for ReLU" story).
- `'kaiming'` — the std stays roughly **constant** across layers. Signal preserved.
  That's the whole goal.

## The implementation

```
def xavier_init(fan_in, fan_out):
    torch.manual_seed(0)
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return torch.round(torch.randn(fan_out, fan_in) * std, decimals=4)

def kaiming_init(fan_in, fan_out):
    torch.manual_seed(0)
    std = math.sqrt(2.0 / fan_in)
    return torch.round(torch.randn(fan_out, fan_in) * std, decimals=4)

def check_activations(num_layers, input_dim, hidden_dim, init_type):
    torch.manual_seed(0)
    x = torch.randn(input_dim)
    stds = []
    for i in range(num_layers):
        fan_in = hidden_dim if i > 0 else input_dim
        fan_out = hidden_dim
        if init_type == 'xavier':
            std = math.sqrt(2.0 / (fan_in + fan_out))
        elif init_type == 'kaiming':
            std = math.sqrt(2.0 / fan_in)
        else:
            std = 1.0                      # plain N(0, 1) "random"
        W = torch.randn(fan_out, fan_in) * std
        x = torch.relu(x @ W.T)            # linear + ReLU
        stds.append(float(x.std()))
    return stds
```

Walkthrough of what it does:

1. `torch.manual_seed(0)` in every function — makes results reproducible (the example
   matrix only matches because of the seed).
2. `std = math.sqrt(2.0 / (fan_in + fan_out))` / `(2.0 / fan_in)` — the two formulas.
   The denominator is the *only* difference between Xavier and Kaiming.
3. `torch.randn(fan_out, fan_in) * std` — `randn` already has std 1, so multiplying
   scales the matrix to exactly the std we want (`torch.round(..., 4)` matches the
   spec's 4-decimal example).
4. `check_activations`: `x` starts as a random input of size `input_dim`; the first
   layer's `fan_in` is `input_dim`, every later layer's is `hidden_dim`, and
   `fan_out` is always `hidden_dim`.
5. `x = torch.relu(x @ W.T)` — build this layer's matrix with the chosen init, then
   forward through `Linear + ReLU`.
6. `stds.append(float(x.std()))` — record how spread out the activations still are;
   the returned list shows exploding vs. stable at a glance.

## Where you'll see this again

- **Dead ReLU detector** (problem 19): a neuron born dead is usually a neuron that
  started with bad weights — init and dead ReLU are the same story from two angles.
- **Training loop / GPT**: modern networks (including GPT) use these same ideas —
  sometimes fancier variants — so training doesn't blow up.
- **Layer / batch / RMS normalization** (problems 12-14): a different family of
  tricks that also fight the "variance drifts through the network" problem.

## Gotchas / common mistakes

- **Mixing up fan_in and fan_out** — `fan_in` = columns (number of inputs), `fan_out`
  = rows (number of outputs). The matrix is built `(fan_out × fan_in)`.
- **Using Xavier for ReLU networks** — the denominator averages in `fan_out`, but ReLU
  kills half the variance anyway; Kaiming's `2/fan_in` is the ReLU-correct version.
- **Forgetting the factor of 2** — `√(1/fan_in)` is neither Xavier nor Kaiming; the
  `2` in Kaiming is the entire compensation for ReLU.
- **Forgetting `torch.manual_seed(0)`** — the example values will never match without
  it, and neither will a re-run of your own experiment.
- **Scaling the wrong way** — multiply `randn` by `std`; dividing would give `1/std`
  and destroy the network.
- **Applying ReLU only sometimes in `check_activations`** — the spec says linear + ReLU
  at *every* layer, including the last.

## One-line summary

Weight initialization sets each layer's starting weights so the variance of activations
neither explodes nor vanishes — Xavier averages both fan sizes for sigmoid, and
Kaiming's extra factor of 2 pays back what ReLU kills.