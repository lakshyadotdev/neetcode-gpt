# 33 — Train Your GPT

## Why this problem exists

Last problem you built a GPT that can do a forward pass. But a model that can't learn is
just a big random number generator. This is the moment everything clicks together: the
gradient-descent loop from problem 01, the softmax + cross-entropy from problems 03-04,
and the GPT from problem 32. "Training" = repeatedly showing the model text, measuring
how wrong its next-token guesses are, and nudging the weights to be less wrong.

This is the "Train" step of the pipeline:

```
Tokenize → Data → Model → Train → Generate
```

## Where this gets used

- Every LLM that has ever been trained runs some version of this loop, on billions of
  tokens instead of one small tensor.
- `optimizer.step()` here is literally gradient descent from problem 01 — just applied
  to millions of weights at once with a fancier step rule (AdamW).
- The cross-entropy here is the same one from problem 04: "how surprised were you by the
  correct next token?"

## The setup

The driver hands you a `MiniGPT` model (it returns raw logits), a 1D integer tensor of
token IDs, plus `epochs`, `context_length`, `batch_size`, and `lr`. You implement the
training loop and return the **final loss, rounded to 4 decimals**.

```
epochs = 5, batch_size = 4, context_length = 4, lr = 0.01
Output: 2.3613
```

The loss should decrease as the model learns.

## Main theory

### What the model is trying to learn

Next-token prediction: given tokens `0...t`, predict token `t+1`. Every position is
simultaneously an input (for predicting the next token) and a target (the thing the
previous position was predicting). This one trick means a plain chunk of text is its own
labeled dataset — no human annotations needed.

### The training loop, step by step

**1. `torch.manual_seed(epoch)`.** Sets the random number generator so each epoch samples
the same batch no matter how many times you rerun — reproducible results.

**2. Sample start indices with `torch.randint`.** We can't feed the whole text at once,
so we pick `batch_size` random starting positions and slice out chunks of
`context_length` tokens each.

**3. Build `X` and `Y` where `Y = X` shifted right by 1.** If `X = [5, 8, 2, 9]`, then the
targets are `Y = [8, 2, 9, ...]` — the next token after each position. Shifting by one is
what turns text into (input, target) pairs.

**4. Forward pass.** `logits = model(X)`, shape `(B, T, C)` where `B` = batch size,
`T` = context length, `C` = vocab size.

**5. Reshape for cross-entropy.** PyTorch's `F.cross_entropy` wants logits as `(N, C)`
and targets as a flat `(N,)`. So flatten logits to `(B*T, C)` and targets to `(B*T)`.
Every position of every sequence becomes one independent prediction.

**6. `loss = F.cross_entropy(logits_flat, targets_flat)`.** This applies softmax internally
and computes how wrong the predictions are. `F.cross_entropy` expects logits — don't
softmax them yourself first.

**7. The backward trio.**

```
optimizer.zero_grad()   # clear last batch's gradients
loss.backward()         # compute gradients of the loss w.r.t. every weight
optimizer.step()        # nudge every weight downhill (gradient descent!)
```

`zero_grad` matters: PyTorch *accumulates* gradients by default, so without clearing
them, batch 2's gradients would be added on top of batch 1's.

### AdamW in one sentence

Adam keeps a per-weight running average of past gradients (momentum) plus a per-weight
step size, and AdamW adds **weight decay** — a gentle force pulling every weight toward
zero so the model doesn't memorize noise. Same `x = x - lr * gradient` idea as problem
01, just with a smarter step.

### Why the loss should drop

Each epoch the model gets more accurate at predicting the next token, so cross-entropy
(average surprise) goes down. A flat or rising loss means something's broken — or the
model is overfitting.

**Production training adds:** gradient clipping (`clip_grad_norm_`) to stop gradients
exploding, cosine learning-rate decay with linear warmup, and validation loss tracking to
detect overfitting.

## The implementation

```
import torch
import torch.nn.functional as F

def train(model, data, epochs, context_length, batch_size, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    final_loss = 0.0
    for epoch in range(epochs):
        torch.manual_seed(epoch)

        # 1. pick batch_size random start positions
        ix = torch.randint(0, len(data) - context_length, (batch_size,))

        # 2. build X (inputs) and Y (targets = X shifted right by 1)
        x = torch.stack([data[i:i + context_length] for i in ix])
        y = torch.stack([data[i + 1:i + context_length + 1] for i in ix])

        # 3. forward pass
        logits = model(x)                       # (B, T, C)

        # 4. flatten for cross-entropy
        B, T, C = logits.shape
        logits_flat = logits.view(B * T, C)
        targets_flat = y.view(B * T)

        # 5. loss + backward trio
        loss = F.cross_entropy(logits_flat, targets_flat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        final_loss = loss.item()

    return round(final_loss, 4)
```

Walkthrough:

1. `torch.randint(0, len(data) - context_length, (batch_size,))` — start indices from
   `0` up to but *excluding* `len(data) - context_length`. Why that bound? We slice
   `data[i+1 : i+context_length+1]` for targets, so the last index used is
   `i + context_length`, which must stay inside `data`. This bound guarantees it.
2. The `torch.stack` lines gather `batch_size` chunks into one `(B, T)` tensor. `Y` is
   `X` shifted one to the right — each position predicts its successor.
3. `logits.view(B*T, C)` flattens the batch and sequence dims into one axis, exactly what
   `F.cross_entropy` expects for its `(N, C)` argument; `y.view(B*T)` gives the matching
   `(N,)` targets.
4. `F.cross_entropy` does softmax + negative log-likelihood in one fused, numerically
   stable step (it can use the log-sum-exp trick, avoiding the `ln(0)` problem from
   problem 04).
5. The backward trio is the whole training story: clear, compute gradients, step.
6. `final_loss = loss.item()` grabs the Python number (loss is a tensor that still holds
   the computation graph). Round only at the very end.

## Where you'll see this again

- **Training diagnostics / dead ReLU detector** (problems 30-31): you need this loop to
  have anything to diagnose.
- **Make GPT Talk Back** (next): once trained, this same loop's model starts generating.
- **Handwritten digit classifier** (problem 31) and every future ML project: same
  zero-grad → backward → step rhythm.

## Gotchas / common mistakes

- **Forgetting `optimizer.zero_grad()`** — gradients accumulate across batches and
  training diverges or stutters.
- **Reshaping wrong** — `F.cross_entropy` needs `(N, C)` logits and flat `(N,)` targets;
  passing the raw `(B, T, C)` tensor silently treats `T` as the class dimension and gives
  nonsense losses.
- **`randint` bound off by one** — picking `i` too close to the end makes `Y` slice past
  the last token (and returns a loss computed on junk).
- **Returning the loss of the wrong epoch / unrounded** — the spec wants the *final* loss
  rounded to 4 decimals.
- **Softmaxing logits before `cross_entropy`** — the function already does it; double
  softmax distorts the loss.
- **Not seeding per epoch** — results become irreproducible and you can't compare runs.

## One-line summary

Training a GPT is just: sample random text chunks, ask the model to predict each next
token, score it with cross-entropy, and nudge all the weights downhill with
zero-grad → backward → step — repeated over and over until the loss stops falling.