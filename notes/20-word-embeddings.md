# 20 — Word Embeddings

## Why this problem exists

A neural network is a machine that does arithmetic on numbers. It has no idea what a
"word" is. If you feed it the string `"cat"`, it literally cannot multiply that by a
weight.

So before any language model can work, words have to become numbers. The naive way is
**one-hot encoding**: make a vector as long as your whole vocabulary, with a single `1`
for your word and `0` everywhere else. With a 100,000-word vocabulary, "cat" becomes a
100,000-dimensional vector that is almost entirely zeros. That's wasteful, and worse, it
encodes *no meaning* — "cat" and "dog" end up exactly as far apart as "cat" and "banana".

**Word embeddings** fix this: each word gets a small, *dense* vector (every entry is a
real number) where similar words land close together. This problem is about what that
lookup actually is under the hood: an embedding is just a row of a matrix.

## Where this gets used

- Every NLP model ever — GPT, BERT, translation, chat — starts by looking up embeddings.
- In PyTorch it's `nn.Embedding(vocab_size, embed_dim)`, and this problem is literally
  what that layer does: a matrix lookup.
- Word2vec (Mikolov et al., 2013) is the classic way embeddings were trained, and its
  famous trick — `king - man + woman ≈ queen` — is how we know embeddings capture meaning.

## The setup

We're handed a pre-initialized embedding matrix and a list of token IDs, and we return
the embedding vectors those IDs point to.

```
embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]   # shape (3, 2): 3 words, 2 dims
token_ids  = [2, 0, 1]

output = [[0.5, 0.6], [0.1, 0.2], [0.3, 0.4]]
```

Token ID `2` picks row `2` (`[0.5, 0.6]`), token ID `0` picks row `0` (`[0.1, 0.2]`),
and so on. No learning here — just the lookup, which is the whole point.

## Main theory

### Why words need vectors at all

Every input to a neural net is a vector (a list of numbers). To handle words, we need a
rule that turns a word into a vector such that *related words produce related vectors*.
That way the patterns the network learns about "king" transfer to "queen", because they
live in the same neighborhood of vector space.

### The embedding table is just a matrix

An embedding table is a matrix of shape `(vocab_size, embed_dim)`:

- one **row** per word in the vocabulary,
- `embed_dim` **columns** (the dimensions of meaning).

To get a word's vector you *index* the matrix: `embedding[token_id]`. The token ID is
just a row number. This is why it's called a **lookup** — there is no math, you're just
grabbing a row.

### What "similar words are close together" means

Each word's vector is a point in `embed_dim`-dimensional space. Similar words are points
that are near each other. "King" and "queen" are near; "king" and "banana" are far.

The standard measure of closeness is **cosine similarity**: the cosine of the angle
between two vectors.

```
cos(a, b) = (a · b) / (|a| |b|)
```

- Same direction → angle 0 → similarity `1` (identical direction).
- Perpendicular → `0` (no relation).
- Opposite → `-1`.

It only cares about *direction*, not length, which is why it's the usual choice for
comparing meaning.

### Vector arithmetic: `king - man + woman ≈ queen`

Once words are vectors, you can do algebra with them. Empirically, the direction from
"man" to "woman" is nearly the same as the direction from "king" to "queen" (royalty is
royalty, the gender is a separate axis). So:

```
king - man + woman ≈ queen
```

You can think of it as: "take 'king', subtract the 'maleness' part, add the 'femaleness'
part." The result lands near "queen". This only works if the embedding space has neatly
organized axes — and it emerges *by itself* from training data, nobody hand-designs it.

### Where do the numbers come from?

The values in the table are **learned** during training, same as weights. Gradients flow
back through the lookup: only the rows that were actually used in the forward pass get
updated. The network pushes words that appear in similar contexts to similar vectors.
That's the entire trick.

## The implementation

```
import numpy as np

def get_embedding(embeddings, token_ids):
    return np.round(embeddings[token_ids], 5)
```

That's the whole solution. Walkthrough:

1. `embeddings[token_ids]` — NumPy fancy indexing. Pass an array of indices and you get
   back the rows at those indices, in that order. `embeddings[[2, 0, 1]]` returns rows
   2, 0, 1. This is exactly `nn.Embedding`'s lookup behavior.
2. The result shape is `(len(token_ids), embed_dim)` automatically — one row per token
   ID, each row being that word's embedding.
3. `np.round(..., 5)` — the spec wants everything rounded to 5 decimal places.

If `token_ids` were a 2D batch, this same line would return shape
`(batch, seq_len, embed_dim)` — which is precisely the shape a transformer consumes.

## Where you'll see this again

- **NLP intro** (problem 21): builds the tokenizer that produces the IDs you look up
  here.
- **Sentiment analysis** (problem 22): first full pipeline — tokenize, embed, average,
  classify. The embedding is the first layer.
- **Positional encoding** (problem 23): adds a position signal *on top of* these same
  embedding vectors.
- **Every GPT problem**: the embedding table is the first layer of the whole model.

## Gotchas / common mistakes

- **Indexing the wrong axis** — you want rows (words), so token IDs index the first
  axis, `embeddings[token_ids]`. Indexing the other way grabs columns, which is garbage.
- **Forgetting the round** — the judge checks 5 decimal places.
- **Expecting math** — there is no matrix multiplication here. It's a *lookup*; only
  later layers do real arithmetic.
- **Confusing token ID with the word itself** — the embedding is the *row*, the ID is
  just the address of that row.

## One-line summary

Word embeddings turn each word into a dense vector by simply looking up a row of a
learned matrix, so that similar words end up close together — and `king - man + woman`
lands near `queen`.