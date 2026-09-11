# 30 — GPT Data Loader

## Why this problem exists

We have a vocabulary (problem 28) that turns text into one long sequence of integers.
But a model can't train on a whole book at once — it needs many small, fixed-size
examples it can shuffle and process in parallel. So we need to slice the sequence into
training-ready pairs.

The key insight — the whole trick of language modeling:

> A language model's training data is just the same sequence shifted by one position.

If the input is `[t1, t2, t3]`, the target is `[t2, t3, t4]`. At every position the
model must predict the next token given everything before it. One sequence of text,
shifted by one, becomes thousands of "guess the next token" examples.

## Where this gets used

- Every autoregressive language model (GPT-2 through modern LLMs) trains on exactly
  this shifted-sequence pattern.
- It's the standard `get_batch` pattern from Karpathy's nanoGPT, and the same idea in
  every LLM training framework.
- It's the bridge between raw tokenized text and the training loop.

## The setup

```
Input:
data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
context_length = 3
batch_size = 2

Output:
X = [[50, 60, 70],
     [40, 50, 60]]
Y = [[60, 70, 80],
     [50, 60, 70]]
```

- `data` — 1D tensor of integer token IDs (the full encoded text).
- `context_length` — how many tokens each training example sees.
- `batch_size` — how many independent examples per batch.

Output is `(X, Y)`, both with shape `(batch_size, context_length)`. For each example
`i`, pick a random start index and take:

- `X[i] = data[start : start + context_length]`
- `Y[i] = data[start + 1 : start + 1 + context_length]`

We call `torch.manual_seed(0)` before generating the random indices so the result is
reproducible.

## Main theory

### What a batch is

A **batch** is a stack of examples the model processes at once. GPUs are fast when
they push many examples through the network simultaneously, so we group them. Shape
`(batch_size, context_length)`: each row is one example, each column is one token
position.

### Why Y is X shifted by one

The model's only job is next-token prediction: given `[10, 20, 30]`, predict
`[20, 30, 40]`. So for a window starting at index `start`:

- `X = data[start : start + C]` is the context — what the model is allowed to see.
- `Y = data[start + 1 : start + 1 + C]` is the answer key — what comes next at each
  position.

At position 0 the model predicts token 1, at position 1 it predicts token 2, and so
on. Same slice, just slid one token to the right. No labels are needed beyond the text
itself — which is why LLMs can train on raw internet text ("self-supervised").

### Why random starts

Each batch element draws a random start index with `torch.randint`. Over many batches
the model sees every part of the text, but each batch is a fresh random sample. That
helps training generalize instead of always replaying the same windows in order.

### Why seed first

`torch.manual_seed(0)` resets the random number generator, so `torch.randint`
produces the same indices on every run — exactly what a test harness (and your own
debugging) needs: deterministic, reproducible batches.

### The example, concretely

With seed 0, `torch.randint(0, 7, (2,))` draws `[4, 3]`. (7 is the number of valid
starts: `len(data) - context_length = 10 - 3`.) So:

- Row 1, start 4: `X = data[4:7] = [50, 60, 70]`, `Y = data[5:8] = [60, 70, 80]`
- Row 2, start 3: `X = data[3:6] = [40, 50, 60]`, `Y = data[4:7] = [50, 60, 70]`

Exactly the expected output.

## The implementation

```python
import torch

def get_batch(data: torch.Tensor, context_length: int, batch_size: int):
    torch.manual_seed(0)                        # reproducible random indices
    ix = torch.randint(0, len(data) - context_length, (batch_size,))

    X = torch.stack([data[i:i + context_length] for i in ix])
    Y = torch.stack([data[i + 1:i + 1 + context_length] for i in ix])
    return X, Y
```

Walkthrough:

1. `torch.manual_seed(0)` — reset the RNG so the random starts are identical on every
   call, matching the test's expectations.
2. `torch.randint(0, len(data) - context_length, (batch_size,))` — draw `batch_size`
   random start positions. The upper bound matters: a window of length
   `context_length` starting at `len(data) - context_length` ends exactly on the last
   token, so every start index yields a full window.
3. `torch.stack([...])` — the list comprehension builds one slice per random start,
   and `stack` glues the rows into a 2D tensor of shape `(batch_size, context_length)`.
4. `Y` uses `i + 1` on both ends: it's the same window as `X`, slid one position
   right. Everything else is identical — which is exactly why it's "just shifted".
5. Return `(X, Y)` — input batch and target batch, ready for the training loop.

## Where you'll see this again

- **Problem 31 (GPT Dataset)**: the same shift-by-one idea, but starting from raw
  text instead of pre-tokenized integers.
- **Training-loop problems**: `get_batch` is called every step; the model predicts on
  `X`, and the loss compares its predictions against `Y`.
- Every real LLM codebase has a version of this function.

## Gotchas / common mistakes

- **Y off by one on the wrong side.** `Y` starts at `i + 1` *and* ends at
  `i + 1 + context_length`. Shift the whole window, not just the start.
- **Wrong upper bound for randint.** A start of `len(data) - context_length + 1`
  would run past the end of the tensor.
- **Forgetting the seed.** Without `torch.manual_seed(0)` the indices change every
  call and the expected output can't be reproduced.
- **Forgetting the batch dimension.** The output must be 2D — `stack` gives that;
  collecting rows in a plain list does not.
- **Data must be a tensor.** Slicing works like lists, but `stack` needs a tensor.

## One-line summary

The GPT data loader = pick random starting points, slice X, slice the same window
shifted one token to the right as Y, and stack them into a batch the model trains on.