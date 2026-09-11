# 02 — Sigmoid & ReLU

## Why this problem exists

A single neuron does two things: it sums up its inputs times weights, then it applies an
**activation function** to that sum. If you skip the activation, no matter how many
layers you stack, the whole network is still just a linear function of the input. Lines
piled on lines are still a line.

That's a problem, because most things we want to learn (images, language, any real
pattern) are *not* linear. Activations inject **non-linearity**, and that's what gives a
network the power to draw curves and separate complex things. Without them a neural
network would be useless.

So this problem exists to get you comfortable with the two most famous activations —
the building blocks every neural network is made of.

## Where this gets used

- Inside **every** neural network, right after each layer's weighted sum.
- **ReLU** is the default activation for hidden layers in basically all modern nets
  (including the transformer / GPT architecture later in this course).
- **Sigmoid** shows up at the very end for binary classification, where you want a
  probability between 0 and 1.
- This is your first taste of "element-wise" math on NumPy arrays, which is how all of
  ML is actually computed.

## The setup

NeetCode gives you a 1D NumPy array `z` and asks you to write two functions that work on
the whole array at once:

1. **Sigmoid**: squishes any input to a value between 0 and 1.
2. **ReLU** (Rectified Linear Unit): returns 0 for negative inputs, the input itself for
   positives.

Both work element-wise — each element of `z` gets its own output at the same position.

## Main theory

### Sigmoid

```
σ(x) = 1 / (1 + e^(-x))
```

The `-x` inside `e` flips the sign, so:

- Very positive input → `e^(-x)` is tiny → denominator near 1 → output near **1**.
- `x = 0` → `e^0 = 1` → `1/(1+1) = 0.5`.
- Very negative input → `e^(-x)` is huge → denominator huge → output near **0**.

So no matter what you feed it, the output is squeezed between 0 and 1. That's why it's
nice to read as a probability. A concrete check: `sigmoid([0]) → [0.5]`, since
`1/(1+e^0) = 1/2`.

**The catch — vanishing gradient.** At the far left and right the curve is basically flat,
so the slope there is nearly zero. In gradient-descent terms, the "push" your weights get
is nearly zero, and learning crawls to a halt. In deep networks that's a real killer, which
is why sigmoid lost popularity for hidden layers.

### ReLU

```
ReLU(x) = max(0, x)
```

Dead simple: positive inputs pass through unchanged, negatives get turned into 0.
Example: `relu([-1, 0, 1]) → [0.0, 0.0, 1.0]`.

**Why it's great:**

- It's dirt cheap to compute (just a `max`).
- For any positive input the slope is exactly 1, so there's **no vanishing gradient** on
  that side — gradients flow through nicely. That's what made training deep networks
  practical, and it's why ReLU is the default everywhere.

**The catch — dead neurons.** If a neuron's input is always negative, it always outputs 0
and its slope is 0, so gradient descent never updates it. It stays "dead" forever.

### Element-wise in NumPy

The point of doing these with NumPy is that you compute the math on the **whole array at
once**, no Python `for` loops. `np.exp(-z)` applies `e^(-z)` to every element in one shot,
and `np.maximum(0, z)` applies the `max` element-wise. "Element-wise" just means each input
gets its own output, same position.

## The implementation

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)
```

Walkthrough:

1. `np.exp(-z)` — computes `e^(-z)` for every element of `z` at once.
2. `1 / (1 + ...)` — the sigmoid formula, applied element-wise.
3. `np.maximum(0, z)` — element-wise max of 0 and each value. `np.maximum` is the
   element-wise version (plain `max` would just give one number).
4. No rounding — these return the raw floats, matching the examples.

## Where you'll see this again

- **Softmax** (problem 03) is a fancier activation that turns raw scores into a
  probability distribution — ReLU inside the network, softmax at the very end.
- Every model from here on (MLP, transformer, GPT) has ReLU (or a cousin) between layers.
- Sigmoid returns for binary classification; the same idea shows up again in
  **cross-entropy loss** (problem 04).

## Gotchas / common mistakes

- **Using `max` instead of `np.maximum`** for ReLU — plain `max` collapses to one number;
  `np.maximum` keeps it element-wise.
- **Looping over the array** instead of using `np.exp` / `np.maximum` — you're supposed to
  do it in one vectorized shot.
- **Wrong sign in sigmoid** — it's `e^(-x)`, not `e^(x)`. (Though `1/(1+e^(-x))` and
  `e^x/(e^x+1)` are the same thing.)
- **Expecting exact 0 or 1** — sigmoid never actually *reaches* 0 or 1, only gets close.

## One-line summary

Activation functions add the non-linearity that lets networks learn complex patterns:
sigmoid squishes to (0, 1) for probabilities, and ReLU (max(0, x)) is the cheap, no-vanishing-gradient default for everything else.