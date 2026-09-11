# 35 — KV-Cache

## Why this problem exists

Your GPT can generate text now — but it's slow. Every new token runs a full forward
pass, and inside each attention layer it recomputes K and V for *all* previous tokens
from scratch. Token 100 wastes most of its time re-deriving keys and values it already
computed at token 50. **KV-Cache** fixes this: store K and V, compute only the new
token's.

It's the single most important inference optimization in production LLMs.

## Where this gets used

- Every time you use ChatGPT, Claude, or any LLM API, KV-Cache is running behind the
  scenes — it's the "Generate" step of the pipeline made fast.
- GQA, paged attention (vLLM), and sliding windows all build on top of it.

## The setup

Implement two things:

1. `KVCache` — a class with `update(new_k, new_v)` that appends new K, V tensors along
   the sequence dimension (`dim=1`) and returns the full cached K, V.
2. `CachedAttention` — a module whose `forward(x, kv_cache)` projects `x` into Q, K, V,
   updates the cache, computes scaled dot-product attention against the *full* cached
   K, V with a causal mask that accounts for cached tokens, then returns the rounded
   output and the updated cache.

`kv_cache` is `None` on the first call; `x` has shape `(batch, seq_len, model_dim)`.

## Main theory

### The problem: quadratic attention

In generation, each step adds one token, then runs a full forward pass:

```
q = W_q @ context      # (1, 10, d)
k = W_k @ context      # (1, 10, d)  <-- recomputed from scratch!
v = W_v @ context      # (1, 10, d)  <-- recomputed from scratch!
scores = q @ k.T       # (1, 10, 10)
output = softmax(scores) @ v
```

At step 10 you compute K and V for tokens 1-9 *again*, even though you already did at
step 9 (and 8, and 7...). The `N × N` attention matrix makes each step `O(N²)` — so
generating 100 tokens recomputes K, V `1 + 2 + ... + 100 = 5050` times total.

Worked example, 4 tokens (length 1 → 4):

| Step | New token | Cache K shape | Work |
|------|-----------|---------------|------|
| 1 | "The" | (1, 1, d) | 1 |
| 2 | " cat" | (1, 2, d) | 2 |
| 3 | " sat" | (1, 3, d) | 3 |
| 4 | " on" | (1, 4, d) | 4 |

Without cache: `1² + 2² + 3² + 4² = 30` operations. With cache: `1 + 2 + 3 + 4 = 10`.

### The key insight

K and V for a token depend only on that token's own embedding and the attention layer's
weights — and neither changes between generation steps. Past tokens stay past. So:

- **Before**: K, V for all `N` tokens every step → `O(N²)` per step.
- **After**: K, V for just the 1 new token, appended to the cache → the new query attends
  over a growing cache → `O(N)` per step.

Only Q is always fresh (the new token asking "who should I look at?"); K and V are the
memory that lets previous tokens answer.

### What the cache stores and the memory tradeoff

The cache holds `2 × L × H × d_h` floats per token (`L` layers, `H` heads, head dim
`d_h`) — K and V for every head per layer. For a 7B-parameter model generating 4096
tokens that's several GB. KV-cache *trades memory for compute*, which is why longer
contexts ("context window costs") need so much memory.

### How attention changes with the cache

Without a cache, the full context sits in `x`, so the attention matrix is `(T, T)` and
the causal mask is "row i can't see columns j > i".

With a cache, the query tensor holds the *new* tokens (`T` of them), but K and V hold
all `cache_len = cached + T` tokens. The mask must shift: a new token at global position
`cached + i` attends to *all* cached tokens plus new tokens up to `i` — so the blocked
region is just the upper triangle of the *new* part.

## The implementation

```
import torch
import torch.nn as nn
import torch.nn.functional as F

class KVCache:
    def __init__(self):
        self.cache_k = None
        self.cache_v = None

    def update(self, new_k, new_v):
        if self.cache_k is None:
            self.cache_k = new_k
            self.cache_v = new_v
        else:
            self.cache_k = torch.cat([self.cache_k, new_k], dim=1)
            self.cache_v = torch.cat([self.cache_v, new_v], dim=1)
        return self.cache_k, self.cache_v


class CachedAttention(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.wq = nn.Linear(model_dim, model_dim)
        self.wk = nn.Linear(model_dim, model_dim)
        self.wv = nn.Linear(model_dim, model_dim)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        q = self.wq(x)                        # (B, T, d)
        k = self.wk(x)                        # (B, T, d)
        v = self.wv(x)                        # (B, T, d)

        if kv_cache is None:
            kv_cache = KVCache()
        k_full, v_full = kv_cache.update(k, v)      # (B, cache_len, d)

        scores = q @ k_full.transpose(-2, -1) / (C ** 0.5)   # (B, T, cache_len)

        cache_len = k_full.shape[1]
        mask = torch.triu(
            torch.ones(T, cache_len, dtype=torch.bool, device=x.device),
            diagonal=cache_len - T + 1,
        )
        scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = attn @ v_full                    # (B, T, d)
        return out.round(decimals=4), kv_cache
```

Walkthrough:

1. `KVCache` is just two growing tensors. `update` concatenates the new K/V onto the
   existing ones **along dim=1** (the sequence dimension) — dim=0 is the batch and must
   not grow. First call (no cache yet) just stores.
2. In `forward`, Q, K, V are projected from the incoming `x` as usual. Then the cache is
   updated — old K/V are reused, only the new token's get appended.
3. `q @ k_full.transpose(-2, -1)` builds the `(T_new, cache_len)` attention matrix, and
   `C ** 0.5` (`sqrt(model_dim)`) is the standard scaling factor.
4. The mask blocks `column > row + (cache_len - T)` — a new token can't peek at later
   new tokens, but every cached token stays visible. Sanity check: first call
   (`cache_len = T`) gives the classic causal mask; a generation step (`T = 1`) blocks
   nothing, since the one new token may see everything cached. ✓
5. `masked_fill` turns blocked cells into `-inf` so softmax zeroes them out.
6. Multiply by the cached `v_full`, round, and hand the cache back alongside.

Sanity check from the spec: three single-token calls produce cache shapes
`(1, 1, 4) → (1, 2, 4) → (1, 3, 4)`, and the third output matches running all three
tokens through attention at once without a cache.

## Where you'll see this again

- **Grouped Query Attention** (next problem): GQA shrinks the cache by sharing K, V
  across groups of heads — it only matters *because* this cache exists.
- **MQA, Paged Attention (vLLM), sliding windows**: production optimizations layered on
  top of KV-cache; every inference framework (nanoGPT, vLLM) implements this pattern.

## Gotchas / common mistakes

- **Concatenating along dim=0 instead of dim=1** — grows the batch instead of the
  sequence; shapes break immediately.
- **Forgetting the `None` first call** — the very first token has no cache yet.
- **Mask off-by-one with cached tokens** — reusing the plain `(T, T)` causal mask when
  tokens are already cached blocks too much (or nothing) and corrupts attention.
- **Not scaling by `sqrt(d)`** — scores blow up and softmax saturates.
- **Rounding inside the cache** — round only the output; rounding stored K/V quietly
  degrades every future step.
- **Recomputing K/V anyway** — the whole point is to reuse the cache; computing K, V for
  all previous tokens each step defeats the optimization.

## One-line summary

KV-Cache stores the K and V of every past token so each new token only computes its own
K and V, turning per-step attention from O(N²) to O(N) — the memory-for-speed tradeoff
that makes LLM generation practical.