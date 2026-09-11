# 32 — Code GPT

## Why this problem exists

For the last several problems you've been building pieces: token embeddings, positional
embeddings, self-attention, transformer blocks. Now it's time to snap them together into
the real thing — a **GPT** (Generative Pre-trained Transformer), the architecture behind
ChatGPT and basically every modern LLM.

This is the "Model" step of the pipeline:

```
Tokenize → Data → Model → Train → Generate
```

The full GPT forward pass is only three lines of math:

```
h0 = TokenEmbed(x) + PosEmbed(positions)
hi = TransformerBlock_i(h_{i-1})   for i = 1 ... N
logits = hN · W_vocab
```

If you can write those three lines, you basically know what an LLM is.

## Where this gets used

- ChatGPT, Claude, Gemini, Llama — they all run this exact forward pass.
- Every piece you've implemented so far (attention, blocks) exists *for* this moment:
  they're the guts of a real GPT.
- Understanding this model makes the next three problems (train it, make it talk, speed
  it up) make sense.

## The setup

You're given `vocab_size`, `context_length`, `model_dim`, `num_blocks`, `num_heads`, and a
`context` — a batch of token-index sequences. The starter code already gives you a working
`TransformerBlock` (from the earlier problem). Your job: write `forward()` that returns
raw **logits** of shape `(batch_size, context_length, vocab_size)`.

```
vocab_size = 5, context_length = 5, model_dim = 16
num_blocks = 4, num_heads = 4
context = [['With', 'great', 'power', 'comes', 'great']]
```

The model internally maps `{'with': 0, 'great': 1, 'power': 2, 'comes': 3,
'responsibility': 4}`.

## Main theory

### The four components of a GPT

**1. Token embedding.** Each token ID becomes a dense vector of size `model_dim`. It's
just a lookup table: `nn.Embedding(vocab_size, model_dim)` with one trainable row per
vocabulary word. "Power" and "great" get their own learned vectors, and words used in
similar places end up with similar vectors.

**2. Positional embedding.** Attention has no built-in notion of order. Without
positional info, "the cat ate the fish" and "the fish ate the cat" would produce the same
hidden states. So we add a second learned lookup table (one row per position, also
`model_dim` wide) and **add** it to the token embedding:

```
h0 = TokenEmbed(x) + PosEmbed(positions)
```

Both have shape `model_dim`, so they add element-wise. This is a *learned* positional
embedding (the original "Attention Is All You Need" paper used fixed sine/cosine waves —
either works, learned is simpler).

**3. Transformer blocks (stacked N times).** Each block is the multi-headed attention +
feedforward + residual connections + layer norm you already built. Stacking blocks lets
the model capture deeper patterns: early blocks handle local grammar, deeper blocks
handle long-range meaning. `num_blocks` is just how many to chain.

```
hi = TransformerBlock_i(h_{i-1})
```

Each block takes the previous output (shape `(B, T, model_dim)`) and returns the same
shape. That's why they can be stacked like Lego.

**4. The output head.** The last linear layer maps from `model_dim` back to `vocab_size`:

```
logits = hN · W_vocab
```

Think of it as scoring every word in the vocabulary against each hidden state: "given
what I've read so far, how much does word #3 fit next?" Higher score = more likely.

### Reading the example logits

Each row `i` of the output is the model's prediction *after* seeing tokens `0...i`:

- Row 0, after "With": `great` has `3.20` — the highest score. Good.
- Row 1, after "With great": `power` gets `4.50`. Good.
- Row 2, after "With great power": `comes` gets `5.00` — the model is confident.
- Row 3, after "With great power comes": the model hedges — `great` `2.80` vs
  `responsibility` `1.20`. Reasonable: "great" could follow, but "responsibility" is
  coming.
- Row 4, after "With great power comes great": `responsibility` jumps to `4.00`. The
  model reconstructed the whole quote.

That's exactly what a well-trained next-token predictor should do.

### Why logits and not probabilities

Logits are raw, unnormalized scores — they can be anything from `-5.00` to `5.00` or
beyond. The softmax (problems 03/04) is applied *later*, during training and generation,
to turn these into probabilities. The spec wants raw logits here, so don't softmax in
`forward()`.

## The implementation

```
class GPT(nn.Module):
    def __init__(self, vocab_size, context_length, model_dim, num_blocks, num_heads):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(context_length, model_dim)
        self.blocks = nn.Sequential(*[
            TransformerBlock(model_dim, context_length, num_heads)
            for _ in range(num_blocks)
        ])
        self.ln_f = nn.LayerNorm(model_dim)
        self.lm_head = nn.Linear(model_dim, vocab_size)

    def forward(self, context):
        B, T = context.shape
        tok = self.token_embedding(context)                  # (B, T, model_dim)
        pos = self.position_embedding(torch.arange(T, device=context.device))
        h = tok + pos                                        # (B, T, model_dim)
        h = self.blocks(h)                                   # (B, T, model_dim)
        h = self.ln_f(h)
        logits = self.lm_head(h)                             # (B, T, vocab_size)
        return logits
```

Walkthrough:

1. `token_embedding(context)` — looks up each token ID, giving `(B, T, model_dim)`.
2. `position_embedding(torch.arange(T, device=...)` — one positional vector per position
   `0..T-1`, shape `(T, model_dim)`. Broadcasting adds it to every batch row. The
   `device=context.device` matters: if your tensor is on GPU, `arange` must match.
3. `h = tok + pos` — the `h0` line of the math.
4. `blocks(h)` — runs all `num_blocks` transformer blocks sequentially.
5. `ln_f(h)` — a final layer norm before the head. Production GPTs do this; it stabilizes
   the hidden states before the big projection.
6. `lm_head(h)` — the `W_vocab` projection. Each position becomes a `vocab_size`-wide
   vector of logits. Done.

Note: `model_dim` must be divisible by `num_heads`, and every sequence must have exactly
`context_length` tokens — the constraints guarantee your shapes work out.

## Where you'll see this again

- **Train Your GPT** (next problem): same model, plus the training loop that learns the
  weights.
- **Make GPT Talk Back**: same model, plus the sampling loop that generates text.
- **KV-Cache / GQA**: same model, plus optimizations that make generation fast.
- Real life: this is nanoGPT's `GPT` class almost verbatim (minus dropout, GELU, and
  fancy weight init).

## Gotchas / common mistakes

- **Forgetting positional embeddings** — the model then can't tell order apart and
  produces garbage.
- **Wrong output shape** — you need `(B, T, vocab_size)`, not `(B, T, model_dim)`.
  The final linear layer is what widens it back to vocabulary size.
- **Applying softmax inside `forward()`** — the spec wants raw logits.
- **Device mismatch** on the position indices — always put `torch.arange` on the input's
  device.
- **Adding instead of concatenating** the token + position embeddings — they must be
  added element-wise (both are `model_dim`), not glued along a dimension.
- **`num_heads` not dividing `model_dim`** — head reshaping will break.

## One-line summary

A GPT is just token embeddings + positional embeddings added together, pushed through N
transformer blocks, and projected out to one score per vocabulary word for every position.