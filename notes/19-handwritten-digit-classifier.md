# 19 — Handwritten Digit Classifier

## Why this problem exists

Everything so far has been pieces: gradient descent, forward passes, normalization,
the training loop, diagnostics. This problem is the first time you assemble them
into a real application — a network that reads handwritten digits 0-9 from images.
The classic MNIST problem.

MNIST is the "hello world" of machine learning: it's the problem where deep
learning first proved it could beat traditional algorithms, and every ML engineer
has solved it. Watch 3Blue1Brown's neural-network videos for the visual intuition
of how these classifiers work.

## Where this gets used

- MNIST is the standard first benchmark in the field — if your pipeline works
  here, the same skeleton scales up to bigger image problems.
- The architecture pattern here — dense layer → ReLU → dropout → dense → sigmoid —
  shows up in real classifiers everywhere.
- The training loop (16) and diagnostics (17-18) you built are exactly what you'd
  use to train and debug this model for real.

## The setup

We build a feedforward network with this pipeline:

```
Input(784) → Linear(512) → ReLU → Dropout(0.2) → Linear(10) → Sigmoid
```

- Each image is 28×28 = 784 pixels, flattened into a vector. A batch arrives with
  shape `(B, 784)` where `B` is the number of images.
- The first `Linear(784, 512)` projects each image into 512 hidden features.
- `ReLU` adds nonlinearity (more on why below).
- `Dropout(0.2)` regularizes during training.
- The final `Linear(512, 10)` produces 10 raw numbers, one per digit.
- `Sigmoid` squashes them into 0-1 confidence values: output `[i]` is "how sure
  the network is that the image shows digit `i`".

The task: define all layers in `__init__` and connect them in `forward()`. No
training, no optimizer. The judge validates the architecture and tensor shapes, not
the actual predictions — your untrained weights will output garbage numbers, and
that's expected.

## Main theory

### Why flatten the image

A `nn.Linear` layer multiplies its input by a weight matrix, and a matrix multiply
expects a vector per sample. So the 28×28 grid gets unrolled into a single
784-element vector. The network never sees the 2D layout — it just sees 784
numbers, and learns which combinations of pixel intensities matter.

### Why ReLU

ReLU is `max(0, x)`: passes positive inputs through untouched, clamps negatives to
0. Two reasons it's the default hidden-layer activation:

- It's cheap to compute, and its derivative is trivially simple (0 or 1).
- It avoids the **vanishing gradient** problem that hurts sigmoid in deep networks
  — a sigmoid's gradient shrinks toward 0 for extreme inputs, so deep layers learn
  slowly. ReLU's gradient stays 1 for positive inputs.

### Why dropout

**Dropout** randomly zeros out a fraction `p` of a layer's activations on *each
training step*. With `p = 0.2`, 20% of the 512 hidden units are turned off per
step. Why is that good? It forces the network not to rely on any single neuron —
it has to spread the knowledge across many, so it generalizes instead of
memorizing. At inference time PyTorch automatically disables dropout, so
predictions use the full network. Dropout has no learnable parameters — it's a
training-time trick, not a transformation of the data.

### Why sigmoid at the end

The last layer must produce a "confidence" per digit, and sigmoid squashes any
number into the range `(0, 1)`. Note the problem deliberately uses sigmoid (not
softmax): the 10 outputs are independent confidences, and the judge only cares
about the `(B, 10)` output shape.

## The implementation

```
import torch.nn as nn

class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(512, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        x = self.linear1(images)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x
```

Walkthrough of what it does:

1. `class DigitClassifier(nn.Module)` — every PyTorch model subclasses
   `nn.Module`, which gives it parameter tracking, `train()`/`eval()` modes, and
   all the machinery PyTorch needs.
2. `super().__init__()` — required boilerplate; without it PyTorch's internals are
   never set up and things break in confusing ways.
3. Each layer is created once in `__init__` and stored as an attribute. That's how
   PyTorch finds the parameters via `model.parameters()`.
4. `forward(self, images)` — connects the layers in the exact order from the spec.
   Trace the shapes flowing through:
   - `(B, 784)` → `linear1` → `(B, 512)`
   - `relu` → `(B, 512)` (shape unchanged, negatives clamped to 0)
   - `dropout` → `(B, 512)` (during training, 20% of values zeroed)
   - `linear2` → `(B, 10)`
   - `sigmoid` → `(B, 10)`, every value in `(0, 1)`
5. The return value matches the expected output in the spec's example: one row of
   10 confidence values per image.

## Where you'll see this again

- **Training loop (16)** — this is the model you'd train with that loop; add a
  loss and an optimizer and it's a complete classifier.
- **Training diagnostics (17)** / **dead ReLU detector (18)** — run these on this
  exact network to check its health while training.
- **GPT (final section)** — the same `nn.Module` pattern (layers in `__init__`,
  wiring in `forward`), just with embeddings and attention instead of pixels.

## Gotchas / common mistakes

- **Forgetting `super().__init__()`** — the most common beginner error; things
  silently break.
- **Wrong dimensions** — `nn.Linear(784, 512)` (input, output) order; the final
  layer must output 10, one per digit.
- **Sigmoid before the last Linear** — the sigmoid goes *after* `linear2`.
- **Dropout in the wrong place or wrong `p`** — the spec wants `nn.Dropout(0.2)`
  between the ReLU and the final Linear, where `0.2` zeroes 20% (it does NOT keep
  20%).
- **Adding training logic** — no optimizer, no loss, no gradient step here; the
  judge only checks the forward pipeline.
- **Expecting sensible predictions** — an untrained model outputs random
  confidences; the judge checks architecture and tensor shape, not values.
- **Reshaping inside `forward`** — the images already arrive flattened as
  `(B, 784)`, so no `view()` is needed.

## One-line summary

The digit classifier is the classic ML hello world: flatten 28×28 pixels into 784
numbers, push them through Linear → ReLU → Dropout → Linear → Sigmoid, and each of
the 10 outputs is the network's confidence that the image shows that digit.