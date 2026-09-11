# 25 — Multi Headed Self Attention

## Why this problem exists

A single attention head can only learn *one* kind of relationship. But a sentence
contains many at once: subject–verb links, adjective–noun pairs, pronoun references,
question words... If the whole `attention_dim` budget goes to one head, that head has to
be mediocre at everything.

Multi-headed attention runs several heads in parallel — each with its own `W_Q`, `W_K`,
`W_V` — and lets each one specialize. One head tracks "which noun does this verb belong
to", another tracks "what does 'it' point at". GPT-3 runs 96 of them at once.

## Where this gets used

- Every transformer: multi-head attention is the standard, single-head was just the
  warm-up.
- The transformer block (problem 26) uses multi-head attention as its first sub-layer.
- Your GPT in the "Build GPT" section uses this exact module.

## The setup

`MultiHeadAttention(embedding_dim, attention_dim, num_heads)`:

- `attention_dim` is now the *total* output size across all heads.
- `num_heads` splits it: each head gets a slice of size `attention_dim // num_heads`, so
  `attention_dim` must divide evenly by `num_heads`.
- `forward(embedded)` takes `(batch_size, context_length, embedding_dim)` and returns
  `(batch_size, context_length, attention_dim)`.

The starter code already gives us `SingleHeadAttention` from problem 24 — we reuse it.

## Main theory

### The formula

```
MultiHead(X) = Concat(head_1, ..., head_h) · W^O

head_i = Attention(X · W_Q^i, X · W_K^i, X · W_V^i)
```

Step by step:

1. **Split the budget.** Each head gets its own slice, size
   `head_dim = attention_dim // num_heads`.
2. **Feed the same input through every head in parallel.** Each head has its own
   `W_Q`, `W_K`, `W_V`, so each learns different queries/keys/values → a different
   attention pattern.
3. **Concatenate** all head outputs along the feature dimension: back to `attention_dim`.
4. **Apply `W^O`**, a learned output projection, to mix information across heads.

### Why multiple heads

One head, one pattern. At initialization all heads are random, and as training proceeds
each one settles into a different "expertise". With more heads you get more simultaneous
views of the sentence, so the model can track subject–verb *and* pronoun-reference at
the same time. That's why GPT-3's 96 heads matter.

### Why concatenate, then project

Each head only ever sees its own slice of the output space — it can't coordinate with
the other heads. Concatenation puts all the slices side by side, and then the learned
`W^O` is a full mixing matrix: it can combine any slice with any other. The output
projection is what lets the model blend the heads' findings into one coherent vector.

### A tiny concrete example

Say `attention_dim = 4`, `num_heads = 2`, so each head works with `head_dim = 2`. For
one token, suppose:

```
head_1 returns  [0.1, 0.9]
head_2 returns  [-0.3, 0.4]
```

Concat: `[0.1, 0.9, -0.3, 0.4]` (four numbers back to back). Then `W^O` (a 4×4 matrix)
mixes them into the final four numbers.

Note the edge case: with `num_heads = 1`, `head_dim = attention_dim`, the concat is a
no-op, and you're back to problem 24. That's exactly what the spec's example shows — it
produces the identical output as the single-head example.

## The implementation

```
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, attention_dim, num_heads):
        super().__init__()
        self.head_dim = attention_dim // num_heads
        self.heads = nn.ModuleList([
            SingleHeadAttention(embedding_dim, self.head_dim)
            for _ in range(num_heads)
        ])
        self.W_O = nn.Linear(attention_dim, attention_dim, bias=False)

    def forward(self, embedded):
        head_outputs = [head(embedded) for head in self.heads]
        concat = torch.cat(head_outputs, dim=-1)
        return self.W_O(concat)
```

Line by line:

1. `self.head_dim = attention_dim // num_heads` — each head's slice size.
2. `nn.ModuleList([...])` — `num_heads` copies of `SingleHeadAttention`, each with
   output size `head_dim`. Must be `ModuleList`, not a plain Python list, so PyTorch
   registers all the heads' weights as parameters (trained, moved to GPU, optimized).
3. `self.W_O = nn.Linear(attention_dim, attention_dim, bias=False)` — the output
   projection; `bias=False` matches the paper's `· W^O`.
4. `head_outputs = [head(embedded) for head in self.heads]` — run every head on the
   same input; each returns `(batch, context, head_dim)`.
5. `torch.cat(..., dim=-1)` — stack the head outputs side by side along the feature
   dim → `(batch, context, attention_dim)`.
6. `self.W_O(concat)` — mix across heads → final `(batch, context, attention_dim)`.

(The spec's example numbers come from the grader's fixed random weights; with
`num_heads = 1` the answer matches problem 24, which is the point of that example.)

## Where you'll see this again

- **Transformer block (26)** uses this as its first sub-layer.
- **code-gpt / train-your-gpt (28–29)**: your model is a stack of blocks containing
  this multi-head module.
- **Grouped query attention (36)** is a memory-saving variant: instead of one
  key/value set per head, groups of heads share K and V.

## Gotchas / common mistakes

- **`attention_dim` not divisible by `num_heads`** — the slice size would be fractional.
  The spec guarantees divisibility, but assert it in your head before dividing.
- **Plain list instead of `nn.ModuleList`** — the heads' weights silently never get
  trained (or moved to GPU / optimizer).
- **Giving each head `attention_dim` instead of `head_dim`** — the concat would blow
  past `attention_dim` and the output shape would be wrong.
- **Forgetting `W^O`** — without the output projection the heads never get mixed.
- **Concatenating on the wrong axis** — must be `dim=-1` (the feature dimension), not
  the context or batch dim.

## One-line summary

Multi-head attention runs several independent attention heads in parallel on slices of
the output space, concatenates their results, and mixes them with a learned projection —
so the model can track many different token relationships at once.