# 36 — Grouped Query Attention

## Why this problem exists

KV-Cache made generation fast, but the cache is still memory-hungry: every head stores
its own K and V for every token. For a model with 64 heads, that's 64 key vectors and 64
value vectors per token per layer. **Grouped Query Attention (GQA)** asks: do we really
need all of them? No — groups of query heads can *share* the same K and V heads.

GQA is the middle ground on a spectrum: standard Multi-Head Attention gives every query
head its own K/V (expensive but best quality), Multi-Query Attention gives all heads one
shared K/V (cheap, slight quality drop), and GQA shares K/V within *groups* of heads —
most of the quality, a fraction of the memory. It's what Llama 2, Llama 3, Mistral, and
Gemma actually use.

## Where this gets used

- **Llama 2 70B**: 64 query heads, 8 KV heads (8x cache reduction).
- **Llama 3**: GQA across all model sizes.
- **Mistral 7B**: 32 query heads, 8 KV heads.
- **Gemma**: GQA as its default attention.
- Everywhere a KV-cache exists and memory is precious — i.e. every modern LLM server.

## The setup

Implement a `GroupedQueryAttention` module whose `forward(x)`:

1. Projects `x` into Q with `num_heads` heads, and K, V with `num_kv_heads` heads.
2. Expands K, V with `repeat_interleave` to match the query head count.
3. Computes scaled dot-product attention with a causal mask.
4. Concatenates heads and applies an output projection.
5. Returns the result rounded to 4 decimals.

Inputs: `model_dim` (divisible by `num_heads`), `num_heads`, `num_kv_heads` (must divide
`num_heads` evenly), and `x` of shape `(batch, seq_len, model_dim)`.

```
model_dim = 8, num_heads = 4, num_kv_heads = 2   # 2 groups, 2 query heads each
x = torch.randn(1, 3, 8) → output shape (1, 3, 8)
```

## Main theory

### The spectrum of attention

| Variant | KV Heads | Cache Size | Quality |
|---------|----------|------------|---------|
| MHA | h | baseline | best |
| GQA | g | g/h of MHA | near-MHA |
| MQA | 1 | 1/h of MHA | slight drop |

`num_kv_heads == num_heads` → GQA reduces to plain MHA. `num_kv_heads == 1` → GQA
reduces to MQA. GQA is the knob you turn to trade memory for quality, in fine steps.

### How KV heads are shared

Standard MHA projects Q, K, V all with `h` heads. GQA projects Q with `h` heads but K
and V with only `g` heads, then **expands** the `g` KV heads back to `h` by repeating
each KV head within its group:

- 8 query heads, 2 KV heads → each KV head serves a group of 4 query heads.
- `repeat_interleave` turns the 2 KV heads into 8 by repeating each one 4 times:
  `[kv0, kv1] → [kv0, kv0, kv0, kv0, kv1, kv1, kv1, kv1]`.
- After expansion, attention runs exactly like standard MHA — scaled dot-product with a
  causal mask.

The 4 query heads in group 0 all attend using `kv0`; the 4 in group 1 use `kv1`. They
see the same keys and values, so they read the same memories — they just combine them
differently through their own Q projections.

### The memory math

With 64 heads and head dim 128, the cache stores `2 × 64 × 128 = 16,384` floats per
token per layer. GQA with 8 KV heads: `2 × 8 × 128 = 2,048` — an **8x reduction**. At
4096 tokens across 80 layers, that's the difference between ~40 GB and ~5 GB of cache
memory. Same math as the table: cache size scales with the number of KV heads.

### Why quality barely drops

Query heads in the same group tend to learn *similar* attention patterns anyway (nearby
heads often specialize in related things). Sharing their K/V loses little, while
halving or dividing the KV head count by 8 saves enormous memory. Empirically GQA lands
very close to MHA quality — close enough that Llama and Mistral bet their production
models on it.

### You can convert an existing MHA model to GQA

The GQA paper's neat finding: take a trained MHA checkpoint, **mean-pool the KV heads
within each group** (average the 4 heads of a group into 1), then fine-tune briefly. You
don't need to retrain from scratch — a converted model recovers near-original quality
fast. That's how most of today's models ended up with GQA.

## The implementation

```
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads
        self.groups = num_heads // num_kv_heads

        self.wq = nn.Linear(model_dim, num_heads * self.head_dim)
        self.wk = nn.Linear(model_dim, num_kv_heads * self.head_dim)
        self.wv = nn.Linear(model_dim, num_kv_heads * self.head_dim)
        self.wo = nn.Linear(num_heads * self.head_dim, model_dim)

    def forward(self, x):
        B, T, C = x.shape

        q = self.wq(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # expand KV heads to match the query heads
        k = k.repeat_interleave(self.groups, dim=1)      # (B, h, T, dh)
        v = v.repeat_interleave(self.groups, dim=1)      # (B, h, T, dh)

        # scaled dot-product attention with causal mask
        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)   # (B, h, T, T)
        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v                                    # (B, h, T, dh)

        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        out = self.wo(out)                                # (B, T, model_dim)
        return out.round(decimals=4)
```

Walkthrough:

1. **Projections.** Q projects to `num_heads × head_dim` (all the query heads), while K
   and V project to `num_kv_heads × head_dim` — the whole memory saving lives in these
   two smaller matrices.
2. **Reshape.** Each projection is split into `(B, T, heads, head_dim)` and transposed to
   `(B, heads, T, head_dim)` so heads become a clean dimension to operate on.
3. **`repeat_interleave(self.groups, dim=1)`** — the core GQA move. Each of the `g` KV
   heads is repeated `groups` times *consecutively*, so query heads `0..groups-1` share
   KV head 0, `groups..2*groups-1` share KV head 1, and so on.
4. **Attention.** `q @ k.transpose(-2, -1)` gives per-head scores, scaled by
   `sqrt(head_dim)` (the per-head scale, not model dim). The `triu` mask with
   `diagonal=1` blocks future tokens; softmax normalizes; `attn @ v` yields head outputs.
5. **Merge and project out.** Transpose heads back to the sequence dim, `.contiguous()`
   (required before `.view()` after a transpose), flatten all heads, and run the final
   `wo` projection back to `model_dim`.
6. Round to 4 decimals, as the spec requires.

Note the key difference from plain MHA: only the shapes of `wk`/`wv` and the
`repeat_interleave` line. Everything after expansion is textbook multi-head attention.

## Where you'll see this again

- Right after KV-Cache: the cache exists, so shrinking it matters — and GQA is exactly
  how Llama/Mistral do it.
- This is the last problem of the track: put it all together and you've built a real,
  modern, cache-efficient GPT — the same recipe as production LLMs.

## Gotchas / common mistakes

- **Using `repeat` instead of `repeat_interleave`.** `repeat` interleaves
  (`[a, b] → [a, b, a, b]`), scrambling groups so a group's query heads no longer share
  one KV head. `repeat_interleave` keeps each head's repeats contiguous
  (`[a, a, b, b]`) — the correct grouping.
- **Projecting K/V with `num_heads` instead of `num_kv_heads`** — that's just MHA again;
  the whole point is smaller K/V projections.
- **Scaling by `sqrt(model_dim)` instead of `sqrt(head_dim)`** — values can be several
  times too large and push softmax to saturation.
- **Forgetting the causal mask** — every token would attend to the future.
- **`.view()` after `.transpose()` without `.contiguous()`** — runtime error or silently
  wrong shapes.
- **`num_kv_heads` not dividing `num_heads`** — groups come out fractional; the
  constraint exists for a reason.

## One-line summary

Grouped Query Attention gives groups of query heads shared K and V heads via
`repeat_interleave`, cutting KV-cache memory by the grouping factor while keeping
near-MHA quality — the trick behind Llama, Mistral, and Gemma.