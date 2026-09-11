# 10 — MLP from Scratch

## Why this problem exists

You've seen a neuron, and you've backpropagated through two layers. This problem
finally stacks neurons into a real network: a **Multi-Layer Perceptron (MLP)**. The
whole thing is just the neuron's forward pass, repeated — the output of one layer
feeds into the next — but the point is bigger than the code:

- A single neuron can only draw a straight boundary between "yes" and "no".
- Stacking layers with ReLU in between lets the network draw curves and learn patterns
  a single neuron never could.
- And this forward pass is *exactly* what backprop (problems 08-09) needs to
  differentiate when you get to training.

So this problem is the "assemble the whole engine" moment of the section. From here on
you're working with real networks, not building blocks.

## Where this gets used

- Everywhere. An image classifier, a spam filter, a language model — all MLPs (or
  fancier cousins of the same idea) under the hood.
- PyTorch's `nn.Sequential(nn.Linear(...), nn.ReLU(), nn.Linear(...))` is literally
  this forward pass.
- The rest of the "Build a Neural Net" section and all of the Training section assume
  you can compute this without thinking.

## The setup

Forward pass through a list of layers, from input to output:

```
hidden layer i:  hᵢ = ReLU(hᵢ₋₁ · Wᵢ + bᵢ)
output layer:    hᵢ = hᵢ₋₁ · Wᵢ + bᵢ        (no activation)
```

- `x` — 1D NumPy array, the input vector.
- `weights` — a list of 2D arrays, one weight matrix per layer.
- `biases` — a list of 1D arrays, one bias vector per layer.
- Output — the final 1D array, rounded to 5 decimal places.

The convention for shapes: each weight matrix is `(previous_size × next_size)`, so a
layer computes `h_prev · Wᵢ` (matrix multiply, not `Wᵢ · h_prev`). In the example,
`weights[0]` is `2×2` (2 inputs → 2 hidden neurons) and `weights[1]` is `2×1`
(2 hidden → 1 output).

### Working the example

`x = [1, 2]`, `weights = [[[0.1, 0.2], [0.3, 0.4]], [[0.5], [0.6]]]`,
`biases = [[0.1, 0.1], [0.0]]`.

```
hidden:  [1, 2] · [[0.1, 0.2], [0.3, 0.4]] + [0.1, 0.1]
       = [1·0.1 + 2·0.3, 1·0.2 + 2·0.4] + [0.1, 0.1]
       = [0.7, 1.0] + [0.1, 0.1] = [0.8, 1.1]
ReLU:    max(0, [0.8, 1.1]) = [0.8, 1.1]        (nothing negative, unchanged)

output:  [0.8, 1.1] · [[0.5], [0.6]] + [0.0]
       = [0.8·0.5 + 1.1·0.6] = [0.4 + 0.66] = [1.06]
```

Final output: `[1.06]`. No ReLU on the output.

## Main theory

### A layer is one matrix multiply

The hidden layer's two neurons are computed in one shot:

```
[0.8, 1.1] = [1·0.1 + 2·0.3,  1·0.2 + 2·0.4] + [0.1, 0.1]
```

Each *column* of the weight matrix is one neuron's weights, and the matrix multiply
runs the dot product for every neuron at once. That's the whole trick of MLPs: layers
are just big dot products, which NumPy (and GPUs) are very fast at.

### Why ReLU (or any non-linearity) between layers

Here's the key idea: a linear function of a linear function is still linear. If you
stack layers with *no* activation, the whole network collapses into one big matrix:

```
(W2 · (W1 · x)) = (W2·W1) · x     ← still just a line
```

That "network" would be no more powerful than the single neuron from problem 07. The
ReLU in between bends each layer's output, so the composition can draw curves. Deep
networks are powerful *because* of the non-linearities, not despite them.

### Why no activation on the output

The last layer emits the raw prediction. For **regression** you want the actual number
(like a price or a score), and any squashing would distort it. (For classification you
would put a softmax there instead — that's a different problem.)

### The shape convention (the one thing to get right)

Each weight matrix is `(previous_size × next_size)` and the layer computes
`h · Wᵢ`. If you instead write `Wᵢ · h`, every matrix in your list needs to be the
transpose and everything breaks. The forward pass is written to match exactly how
`weights` is handed to you.

## The implementation

```
def forward(x, weights, biases):
    h = np.array(x)
    for i in range(len(weights)):
        h = np.dot(h, weights[i]) + biases[i]
        if i < len(weights) - 1:      # hidden layers only
            h = np.maximum(0, h)      # ReLU
    return np.round(h, 5)
```

Walkthrough of what it does:

1. `h = np.array(x)` — the "activation" starts as the input vector.
2. The loop runs once per layer: `np.dot(h, weights[i]) + biases[i]` is the linear
   part — multiply the current activations by this layer's weights, add the bias.
3. `if i < len(weights) - 1` — apply ReLU to every layer **except the last one**.
   `np.maximum(0, h)` is ReLU, element-wise.
4. `np.round(h, 5)` — round the final activations to 5 decimals and return.

Note: we round only at the very end. Rounding intermediate activations would quietly
corrupt the later layers, and NumPy is exact enough that you don't need to.

## Where you'll see this again

- **Weight initialization** (problem 11): now that you can build the network, the next
  question is where its weights *start*.
- **Training loop** (problem 15): this forward pass plus the backprop from problems
  08-09 is the complete training cycle.
- **Handwritten digit classifier** (problem 19): an MLP is exactly the model you'll
  train there.
- **GPT**: a transformer block is a fancier layer stack, but the forward pass is the
  same pattern — activations flowing through matrix multiplies.

## Gotchas / common mistakes

- **Applying ReLU to the output layer** — the spec's example (`[1.06]`) is positive so
  it survives a stray ReLU, but the last layer must stay linear for regression.
- **Writing `W·h` instead of `h·W`** — the weight matrices are given as
  `(prev × next)`; multiplying in the wrong order needs transposes everywhere.
- **Forgetting the bias** — every layer adds its bias vector after the multiply.
- **Stacking without activations** — a "deep" network with no ReLU is mathematically
  just one linear layer; it's the ReLU that makes depth meaningful.
- **Rounding inside the loop** — round the final answer only.
- **Not rounding to 5 decimals** — NeetCode compares rounded values.

## One-line summary

An MLP is just repeated `h = h·W + b` with a ReLU between hidden layers (and none on
the output) — the same neuron from problem 07, stacked — and this forward pass is what
every neural network library computes for you.