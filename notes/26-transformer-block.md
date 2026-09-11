# 26 — Transformer Block

## Why this problem exists

Attention alone doesn't make a language model — real ones (GPT-2 with 12 blocks, GPT-3
with 96) are stacks of identical **transformer blocks**. Each block does two jobs:

1. **Multi-head attention** — tokens swap information (problems 24–25).
2. **Feed-forward network** — each token "thinks" about what it just gathered.

Wrapped around both: **residual connections** and **layer normalization**. These two
tricks are why you can stack 96 blocks without training collapsing into mush. This block
is *the* unit that all of "Build GPT" repeats.

## Where this gets used

- GPT-2: 12 blocks. GPT-3: 96. Every modern LLM is a stack of these.
- This is decoder-only (GPT-style), so we skip the encoder's cross-attention sub-layer.
- Everything in "Build GPT" — `code-gpt`, `train-your-gpt`, `kv-cache`,
  `grouped-query-attention` — operates inside this block.

## The setup

`TransformerBlock(model_dim, num_heads)`:

- `model_dim` — one size used everywhere: embeddings in, attention out, block out.
- `num_heads` — heads for the multi-head attention inside (`model_dim` must divide
  evenly by it).
- `forward(embedded)` takes `(batch_size, context_length, model_dim)` and returns the
  same shape.

The block is defined by two equations:

```
x = x + MultiHeadAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

## Main theory

### The two sub-layers

- **Multi-head attention** lets every token look at every earlier token and pull in
  relevant context (problems 24–25).
- **Feed-forward** is a two-layer MLP applied to *each token position independently* —
  it never mixes tokens. Attention gathers context; the feed-forward is where the model
  reasons over it. The standard size: expand `model_dim` → `4 · model_dim`, apply an
  activation, project back to `model_dim`. The original paper used ReLU; GPT uses GELU.

### Residual connections

`x + SubLayer(x)` — the sub-layer's output is *added* to the input, not replacing it.

Why it matters: backpropagation sends the error signal backward through the network, and
in deep stacks that signal keeps getting multiplied and can shrink toward zero (the
**vanishing gradient** problem) — early layers learn nothing. A residual connection
gives the gradient a "highway": the derivative of `x + f(x)` with respect to `x` is
`1 + ...`, so there's always a clean `1` path for the gradient to flow through
unchanged, no matter how deep. Bonus: at initialization the sub-layers output near-zero,
so the block starts as close to a no-op — training begins from a sane place.

### Layer normalization (LayerNorm)

Before each sub-layer, normalize each token's vector along the *feature* dimension:
subtract the mean, divide by the standard deviation, then scale and shift by two learned
parameters (γ and β). Now every vector entering a sub-layer has a stable scale (zero
mean, unit variance-ish). Activations don't drift into extreme ranges, so training stays
stable and converges faster.

### Pre-LN vs Post-LN

- Original paper: **Post-LN** — `LayerNorm(x + SubLayer(x))` (normalize after).
- Modern GPT: **Pre-LN** — `x + SubLayer(LayerNorm(x))` (normalize before).

The spec uses Pre-LN. It trains more stably, and it's what GPT-2/GPT-3 actually do.

### Shapes line up

Multi-head attention maps `model_dim → model_dim`, and the feed-forward maps
`model_dim → model_dim`. So `x + SubLayer(x)` is adding two identically-shaped tensors —
element-wise addition just works. That's why the block can be stacked forever: same
shape in, same shape out.

## The implementation

```
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.attention = MultiHeadAttention(model_dim, model_dim, num_heads)
        self.norm2 = nn.LayerNorm(model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Linear(4 * model_dim, model_dim),
        )

    def forward(self, embedded):
        x = embedded
        x = x + self.attention(self.norm1(x))     # sub-layer 1
        x = x + self.feed_forward(self.norm2(x))  # sub-layer 2
        return x
```

Line by line:

1. `self.norm1` / `self.norm2` — the two LayerNorms, one before each sub-layer.
2. `MultiHeadAttention(model_dim, model_dim, num_heads)` — attention whose output width
   equals `model_dim`, so the residual add works.
3. `nn.Sequential(...)` — the feed-forward: expand to `4 · model_dim`, `GELU`, squash
   back to `model_dim`.
4. `x = x + self.attention(self.norm1(x))` — the Pre-LN residual: normalize first,
   attend, then *add the original x back*. Note the order: norm inside, `x +` outside.
5. `x = x + self.feed_forward(self.norm2(x))` — same pattern for sub-layer 2.
6. `return x` — same shape as the input: `(batch, context, model_dim)`.

(The spec's example output numbers come from the grader's fixed random weights; the
example's real point is that the block keeps the input shape.)

## Where you'll see this again

- **code-gpt (28) / train-your-gpt (29)**: your GPT is a stack of these blocks followed
  by a final projection to vocabulary scores.
- **kv-cache (34)**: speeds up the attention *inside* this block.
- **grouped-query attention (36)**: a cheaper attention head used inside this block.

## Gotchas / common mistakes

- **Pre-LN order** — it's `x + SubLayer(LayerNorm(x))`, not `LayerNorm(x + SubLayer(x))`.
  Putting the norm on the outside is Post-LN and changes training behavior.
- **Forgetting the residual `x +`** — returning just `SubLayer(x)` drops the gradient
  highway, and the block no longer starts as a no-op.
- **Feed-forward hidden size** — the standard expansion is `4 · model_dim`; that's what
  GPT and the grader use.
- **Activation choice** — use `GELU`, not the paper's original `ReLU`. The grader's
  expected output depends on it.
- **`model_dim` not divisible by `num_heads`** — same divisibility rule as problem 25.
- **Adding mismatched shapes** — attention output must be `model_dim` for the residual
  add to work.

## One-line summary

A transformer block is attention + feed-forward, each wrapped as `x + SubLayer(LayerNorm(x))`
— residuals keep gradients alive, layer norm keeps training stable, and stacking dozens
of these blocks *is* a GPT.