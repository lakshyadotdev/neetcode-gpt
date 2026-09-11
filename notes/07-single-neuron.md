# 07 — Single Neuron

## Why this problem exists

You now know gradient descent, activations (sigmoid and ReLU), and linear regression.
This problem puts them together into the smallest real building block of neural
networks: the **neuron** (also called a **perceptron**).

A neuron is basically a mini-version of the linear regression model from before
(weighted sum + bias), but with one addition: the result gets squeezed through an
**activation function**. That one addition is what makes stacking many neurons
powerful — without it, a stack of linear functions is still just one linear function.

This problem is deliberately tiny. Its whole job is to make the single-neuron math
automatic so the next problems (backprop, MLPs) don't have to explain it again.

## Where this gets used

- **Every neural network** — image classifiers, GPT, everything — is a stack of
  neurons. Learn this unit and you know what the whole network is doing at the bottom.
- The "Build a Neural Net" section (problems 07-11) is literally this neuron, then
  making it learn, then stacking it into layers.
- PyTorch's `nn.Linear` layer does exactly this: `x @ W.T + b` followed by whatever
  activation you add on top.

## The setup

We implement one function, `forward()`, that computes:

```
output = activation(Σᵢ wᵢ·xᵢ + b)
```

- `x` — the input values (a 1D array).
- `w` — the weights, one per input (same length as `x`).
- `b` — a single bias number.
- `activation` — the string `"sigmoid"` or `"relu"`.
- Output — a single float, rounded to 5 decimal places.

The two activations you need:

```
sigmoid(z) = 1 / (1 + e^(-z))     →  squishes any number into (0, 1)
relu(z)    = max(0, z)            →  clips negatives to zero
```

## Main theory

### The weighted sum (dot product)

`Σᵢ wᵢ·xᵢ` is a **dot product**: multiply each input by its weight, then add them all
up. Think of each input as *voting* with a strength of `wᵢ`. A weight near zero means
"ignore this input"; a big positive or negative weight means "this input matters a lot".

### The bias

The bias `b` is added after the weighted sum. It's the neuron's *baseline tendency*:
even when every input is zero, the neuron still starts from `b`. Changing the bias
shifts the point at which the neuron "wakes up" (for ReLU, the point where it stops
outputting zero; for sigmoid, the midpoint of the S-curve).

### The activation

The activation is applied **after** the weighted sum, never to individual inputs.
Its job is to make the neuron's output non-linear:

- **Sigmoid** maps to a smooth (0, 1) range. It reads like a probability or a "firing
  strength". Historically the default; its downside is that far from 0 it goes flat,
  which makes learning slow (you'll meet this as "saturation" in backprop).
- **ReLU** is just `max(0, z)`. It's fast and it doesn't flatten out on the positive
  side. Its catch: negative inputs give output exactly 0 and gradient 0 — which causes
  "dead neurons" later.

Why non-linearity matters at all: a weighted sum of inputs is a line. A line of a line
is still a line. Only by bending the signal (with ReLU/sigmoid) does a network gain the
ability to draw curves and learn patterns that aren't straight.

### Working the examples

Example 1 — sigmoid: `x = [1, 2]`, `w = [0.5, 0.5]`, `b = 0`.

```
z = 0.5·1 + 0.5·2 + 0 = 1.5
σ(1.5) = 1 / (1 + e^(-1.5)) = 0.81757
```

Example 2 — relu: `x = [1, -2, 3]`, `w = [0.1, 0.2, 0.3]`, `b = -0.5`.

```
z = 0.1·1 + 0.2·(-2) + 0.3·3 + (-0.5) = 0.1 - 0.4 + 0.9 - 0.5 = 0.1
ReLU(0.1) = 0.1
```

Note the second example: `b = -0.5` is enough to pull a `0.6` weighted sum down to
`0.1` — the bias really shifts the output.

## The implementation

```
def forward(x, w, b, activation):
    z = np.dot(w, x) + b
    if activation == "sigmoid":
        out = 1 / (1 + np.exp(-z))
    elif activation == "relu":
        out = max(0, z)
    return round(float(out), 5)
```

Walkthrough of what it does:

1. `np.dot(w, x)` — the weighted sum; a dot product does `Σ wᵢ·xᵢ` in one line.
2. `+ b` — add the bias.
3. Sigmoid branch — `1 / (1 + np.exp(-z))` is the sigmoid formula, and `np.exp(-z)`
   is a stable way to write `e^(-z)`.
4. ReLU branch — `max(0, z)` clips negatives to zero.
5. `round(float(out), 5)` — NeetCode expects the answer rounded to 5 decimals. The
   `float()` just makes the numpy number a plain Python float.

## Where you'll see this again

- **Backpropagation** (problem 08): the *backward* pass of this exact neuron — how
  each weight caused the error.
- **MLP from scratch** (problem 10): this neuron repeated across layers, with ReLU
  between them and no activation on the output.
- **Everything in PyTorch**: `nn.Linear` + an activation is a neuron; stacks of these
  are every modern network.

## Gotchas / common mistakes

- **Forgetting the bias** — the weighted sum alone is only half the neuron.
- **Applying the activation to each input** — it goes on the *sum*, once.
- **Rounding the wrong thing** — round the final output, not the pre-activation.
- **Getting the dot product backwards** — `np.dot(w, x)` where the weights come first
  and inputs second (or vice versa, as long as the shapes match); the order here
  doesn't change the math but keep it consistent.
- **Confusing the two activations** — sigmoid squishes to (0, 1); ReLU *clips to
  zero*. ReLU never squishes anything.

## One-line summary

A neuron is just a weighted sum of its inputs plus a bias, pushed through a
non-linear activation — and every neural network in this course is neurons like this,
stacked.