# 23 — Positional Encoding

## Why this problem exists

RNNs read text one word at a time, left to right, so the model always knows what came
before what. Transformers threw that away: they process **all tokens in parallel**. That
makes them fast, but it costs them all sense of order. To a transformer with no extra
help, the sentence "dog bites man" is literally identical to "man bites dog" — the same
bag of words.

So we need to inject "where am I in the sentence?" into the model. **Positional encoding**
does exactly that: it adds a position signal to each word's embedding, and it does it
using sine and cosine waves at different frequencies. This is the original trick from the
*Attention Is All You Need* paper, and it's the first piece of the transformer you'll
build.

## Where this gets used

- The original Transformer (machine translation) adds these sinusoidal encodings directly
  on top of the word embeddings.
- Modern models like GPT use *learned* positional embeddings instead — same idea, but the
  model trains the position vectors rather than using fixed formulas.
- The sinusoidal version still matters: it generalizes to sequence lengths the model
  never saw in training, which fixed learned tables can't do.

## The setup

We're given `seq_len` (number of positions) and `d_model` (dimension of the encoding,
always even). We must produce a `(seq_len, d_model)` array of position vectors, rounded
to 5 decimal places.

For position `pos` and dimension index `i`:

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

Even dimensions get sine, odd dimensions get cosine, and both share the same "angle"
inside.

## Main theory

### What the formulas mean, in plain words

Each position `pos` gets its own vector of length `d_model`. For each *pair* of
dimensions `(2i, 2i+1)` there's an angle:

```
angle(pos, i) = pos / 10000^(2i / d_model)
```

and the pair stores `(sin(angle), cos(angle))`. Different pairs use different angles
because the exponent `2i / d_model` changes the frequency. So the encoding is really a
bunch of `(sin, cos)` pairs, each oscillating at its own speed.

### Work the example by hand

Spec example: `seq_len = 2, d_model = 4`. There are two pairs (`i = 0` and `i = 1`).

**Position 0.** Every angle is `0 / anything = 0`. And `sin(0) = 0`, `cos(0) = 1`:

```
PE(0) = [sin(0), cos(0), sin(0), cos(0)] = [0, 1, 0, 1]
```

That's why position 0 is always `[0, 1, 0, 1, ...]` for any `d_model`.

**Position 1.**
- Pair `i = 0`: exponent `2·0/4 = 0`, so the angle is `1 / 10000^0 = 1 / 1 = 1`.
  `sin(1) = 0.84147`, `cos(1) = 0.5403`.
- Pair `i = 1`: exponent `2·1/4 = 0.5`, so the angle is `1 / 10000^0.5 = 1 / 100 = 0.01`.
  `sin(0.01) = 0.01`, `cos(0.01) = 0.99995`.

```
PE(1) = [0.84147, 0.5403, 0.01, 0.99995]
```

Full output matches the spec:

```
[[0.0, 1.0, 0.0, 1.0],
 [0.84147, 0.5403, 0.01, 0.99995]]
```

Notice the story the numbers tell: the first pair (`i=0`) swung dramatically from
`(0, 1)` to `(0.84, 0.54)` — fast frequency, good at distinguishing *nearby* positions.
The second pair (`i=1`) barely moved (`(0, 1)` → `(0.01, 1.0)`) — slow frequency, good at
signaling broad position. Every position gets a unique combination, like a fingerprint.

### Why sine and cosine specifically?

Two reasons, both from the paper.

**Reason 1: relative positions become linear.** For a fixed offset `k`, trig identities
give:

```
sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
```

So `PE(pos + k)` is a linear combination of `PE(pos)`. That means the model can *learn*
"two tokens apart" as one simple pattern that looks the same at any absolute position —
no matter whether it's positions 3 and 5 or positions 40 and 42. The model can attend to
relative distance without extra machinery.

**Reason 2: the barcode effect.** Using frequencies spaced exponentially (wavelengths
from `2π` up to `10000·2π`) is like binary: a small number of bits uniquely identifies a
big range of numbers. Each position's full vector is a unique "barcode" of fast and slow
oscillations, so the model can tell every position apart.

### How it's combined with embeddings

You don't concatenate — you **add**: `embedding + positional_encoding`. Same shape, so
addition is free, and the embedding's semantic content and the position signal just share
the vector. The model can pull them apart during training.

## The implementation

```
import numpy as np

def positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    positions = np.arange(seq_len)[:, None]              # (seq_len, 1)
    i = np.arange(d_model // 2)                          # pair indices 0..d_model/2-1
    angles = positions / np.power(10000.0, (2 * i) / d_model)   # (seq_len, d_model/2)
    pe[:, 0::2] = np.sin(angles)     # even dims: sine
    pe[:, 1::2] = np.cos(angles)     # odd dims:  cosine
    return np.round(pe, 5)
```

Walkthrough:

1. **`np.zeros((seq_len, d_model))`** — the output canvas; one row per position.
2. **`np.arange(seq_len)[:, None]`** — positions `0..seq_len-1` reshaped to a column
   vector so broadcasting works (see step 4).
3. **`np.arange(d_model // 2)`** — the pair index `i`. There are `d_model/2` pairs because
   each pair uses two dimensions.
4. **`angles`** — this is the clever part. Dividing the `(seq_len, 1)` positions by the
   `(d_model/2,)` powers broadcasts into a `(seq_len, d_model/2)` matrix: each position
   paired with every pair-index frequency. Each column is one frequency.
5. **`pe[:, 0::2]` / `pe[:, 1::2]`** — fancy slicing fills every even column with the
   sines and every odd column with the cosines. One vectorized line does all positions
   and all dimensions at once.
6. **`np.round(..., 5)`** — spec requires 5 decimal places.

## Where you'll see this again

- **Self-attention** (problems 25-26): this encoding is what lets attention know the
  order of the words it's mixing together.
- **Transformer block** (problem 27): embeddings + positional encodings are the input to
  the whole block.
- **GPT problems**: GPT uses learned position vectors, but the "add position info to the
  embedding" design is unchanged.
- **KV-cache / grouped-query attention**: all built on this same token + position input.

## Gotchas / common mistakes

- **Forgetting `d_model` is even** — the pair scheme needs two columns per pair. The spec
  guarantees it, but your code assumes it.
- **Using `i` instead of `2i` in the exponent** — the formula divides by
  `10000^(2i/d_model)`, not `10000^(i/d_model)`. Wrong exponent = wrong frequencies.
- **Positions starting at 1** — positions start at `0`, which is why row 0 is all
  `[0, 1, 0, 1, ...]`.
- **Sin/cos in degrees** — NumPy's `np.sin`/`np.cos` take *radians*. Your angles are
  plain numbers (radians), so that's fine — just don't convert.
- **Forgetting the round** — 5 decimal places, checked exactly.
- **Swapping sine and cosine columns** — even dims get sine, odd dims get cosine.

## One-line summary

Positional encoding stamps each word with a unique position fingerprint by adding
`(sin, cos)` pairs at exponentially spaced frequencies to its embedding, letting the
order-blind transformer tell "dog bites man" from "man bites dog".