# 24 — Self Attention

## Why this problem exists

"The animal didn't cross the street because **it** was too tired." Which word is "it"?
You know instantly — the animal. But a model that just averages word embeddings (the
Bag of Words shortcut from sentiment analysis) can't answer. It threw away every
relationship between words.

Self-attention fixes that. It lets each token "look at" every other token, decide which
ones matter, and pull in their information — "it" learns to look hard at "animal" and
mostly ignore "street". This one mechanism is the engine of every modern language model.

## Where this gets used

- Every transformer ever built — GPT, BERT, all of them. The formula is from the paper
  "Attention Is All You Need" (Vaswani et al., 2017).
- Problems 25 and 26 build on this: multi-head runs several in parallel; the transformer
  block wraps one in residuals and layer norm.
- The whole "Build GPT" section is this code, plus the machinery around it.

## The setup

We write a PyTorch module called `SingleHeadAttention`:

- `embedding_dim` — the size of each token's input vector.
- `attention_dim` — the size of the output vector produced for each token.
- `forward(embedded)` takes `(batch_size, context_length, embedding_dim)` and returns
  `(batch_size, context_length, attention_dim)`.

`context_length` = the number of tokens in the sequence. And because this is GPT-style
(autoregressive) attention, we also need a **causal mask**: token `i` may only look at
tokens `0..i`, never the future.

## Main theory

### Q, K, V: the library analogy

Each token gets projected into three vectors using three learned weight matrices:

```
Q = X · W_Q      K = X · W_K      V = X · W_V
```

- **Query (Q)** — what this token is searching for (a library patron's question).
- **Key (K)** — what each token is labeled with (the index cards / book labels).
- **Value (V)** — the actual information in the token (the book off the shelf).

Matching a query against the keys tells you *which* tokens matter; the values are the
*content* you blend in. All three come from the same input `X` — that's why it's called
**self**-attention: the sequence attends to itself.

### The attention formula

```
Attention(Q, K, V) = softmax( Q · Kᵀ / √d_k ) · V
```

Four steps:

1. `Q · Kᵀ` — dot every query against every key. Big score = that pair is compatible =
   token A has something to say to token B. Result: a `(context, context)` score grid.
2. Divide by `√d_k` — keep the numbers from exploding.
3. `softmax(...)` — turn each row of scores into weights that are positive and sum to 1.
4. `... · V` — replace each token's vector with a weighted blend of all the values.

### Why divide by √d_k

Dot products grow with dimension: if `d_k` = 512, a dot product is a sum of 512 terms
and gets large; feed big numbers into softmax and it saturates — one weight goes to 1,
the rest to 0, and the gradients become tiny. The fix is statistical: if each dimension
has variance ~1, the dot product has variance ~`d_k`, so its standard deviation is
`√d_k` — dividing by it pulls scores back to a range where softmax stays "soft".

### Why softmax

Softmax turns a row of raw scores into a probability distribution — positive weights
that sum to 1: a 40% / 60% split of how much of each token to blend in, not raw scores
like 1.2 and 3.7. It also exaggerates the winner, so the most relevant token gets the
biggest weight (you met it in problem 03).

### The causal mask

GPT generates one token at a time, so at position `i` it must not peek at positions
`i+1, i+2, ...` — that would be cheating. Before softmax we set all future positions to
`-infinity`; then `e^(-inf) = 0`, so softmax hands them a weight of exactly 0 and the
token only blends in itself and the past.

### Self-attention ignores order (why you needed positional encoding)

If you shuffle the tokens, self-attention gives the *same* attention pattern — the math
doesn't care about position. That's why problem 23 (positional encoding) came first: it
stamps each token with where it sits, so attention *can* use order.

### A concrete walk-through (fake weights on purpose)

Real weights are random and learned; for a hand-calc we'll pretend `W_Q = W_K = W_V`
are identity matrices, so `Q = K = V = X`. Two tokens, `d_k = 2`:

```
token 0: [0.5, -0.2]      token 1: [0.1, 0.8]
```

Step 1 — scores `Q · Kᵀ` (row = query token, column = key token). Each entry is a dot
product; token 1 matches itself best:

```
[[0.29, -0.11],
 [-0.11,  0.65]]
```

Step 2 — divide by `√2 ≈ 1.41`, mask the future (`-inf`), softmax each row, then blend
each token's weights with V:

```
row 0: [0.21,  -inf]   -> weights [1,   0  ]   -> output [0.50, -0.20]
row 1: [-0.08, 0.46]   -> weights [0.37, 0.63] -> output [0.25,  0.43]
```

Token 1 is "mostly itself, a bit of token 0"; token 0 is frozen (mask) since nothing
came before it. That's the entire mechanism.

## The implementation

```
import torch
import torch.nn as nn

class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim, attention_dim):
        super().__init__()
        self.W_Q = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.W_K = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.W_V = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded):
        Q, K, V = self.W_Q(embedded), self.W_K(embedded), self.W_V(embedded)

        scores = Q @ K.transpose(-2, -1) / (K.shape[-1] ** 0.5)

        context_length = embedded.shape[1]
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        return weights @ V
```

Line by line:

1. `nn.Linear(embedding_dim, attention_dim, bias=False)` — the three learned projection
   matrices `W_Q`, `W_K`, `W_V`; no bias keeps it exactly `X · W`.
2. `Q @ K.transpose(-2, -1)` — swaps only the last two dims (context ↔ feature) so the
   matmul gives `(batch, context, context)` scores. `.T` would flip *all* dims — wrong.
3. `/ (K.shape[-1] ** 0.5)` — the `√d_k` scaling (`K.shape[-1]` is `d_k`).
4. `torch.triu(ones(L, L), diagonal=1).bool()` — upper triangle *excluding* the
   diagonal, i.e. `True` where column > row (the future).
5. `masked_fill(mask, float("-inf"))` — `-inf` in the future, *before* softmax.
6. `torch.softmax(scores, dim=-1)` — softmax over the key dimension (each row) so each
   token's weights sum to 1.
7. `weights @ V` — the weighted blend → `(batch, context, attention_dim)`.

## Where you'll see this again

- **Multi-headed self attention (25)**: several of these heads in parallel, each
  watching for a different pattern.
- **Transformer block (26)**: this head wrapped in residuals and layer norm.
- **Everything in Build GPT**: `code-gpt` and `train-your-gpt` are stacks of blocks
  built on this exact code; KV-cache (34) and grouped-query attention (36) optimize it.

## Gotchas / common mistakes

- **Scaling in the wrong place** — divide by `√d_k` before softmax or you lose the
  protection.
- **Mask after softmax** — the `-inf` must be inside the softmax; masking the final
  weights breaks the math.
- **`diagonal=1`, not 0** — the diagonal must stay visible (a token attends to itself).
- **`.transpose(-2, -1)` not `.T`** — you only want to swap the last two dimensions.
- **`bias=False`** — the formula is `X · W`; a bias term changes the projection.

## One-line summary

Self-attention turns each token into a query and everyone else into keys and values,
scores every pair with scaled dot products, softmaxes those scores into blend weights,
and replaces each token with a weighted mix of the values — the "it → animal" lookup
that lets language models understand context.