# 22 — Sentiment Analysis

## Why this problem exists

Can a computer read "this movie was amazing" and "waste of two hours" and tell which is
happy and which is angry? That's **sentiment analysis**, and it was one of the first NLP
tasks deep learning genuinely crushed. For you, though, the point is bigger: this is your
*first complete NLP pipeline*. Everything you've built so far — embeddings, tokenization
— finally gets wired together into one end-to-end model, and you get to write the whole
thing yourself.

## Where this gets used

- Product review classification, social media monitoring, support ticket triage — any
  place that needs "is this text positive or negative?"
- The **Bag of Words** trick here (average all the word vectors, then classify the
  average) is the foundation for much bigger ideas. Transformers do the same embedding →
  aggregate → classify shape, just with a much smarter aggregation (attention).
- The architecture you'll write — `Embedding → mean → Linear → Sigmoid` — is a real
  template you'll recognize everywhere.

## The setup

We write the model skeleton only — `__init__` (define the layers) and `forward` (run the
data through). No training logic. The architecture is fixed:

```
Embedding(V, 16) → mean(dim=1) → Linear(16, 1) → Sigmoid
```

- `Embedding(vocabulary_size, 16)`: maps each token ID to a learned 16-dimensional vector.
- `mean(dim=1)`: averages all the word vectors in a sentence into one 16-dim vector.
- `Linear(16, 1)`: one neuron squashing that vector into a single number.
- `Sigmoid`: turns that number into a probability in `[0, 1]` — near 0 = negative, near 1 = positive.

Input is an integer token-ID tensor of shape `(batch_size, sequence_length)`; output is
`(batch_size, 1)`.

## Main theory

### The pipeline, in order

A sentence comes in as a row of token IDs, e.g. `[2, 7, 14, 8, 0, 0, ...]`. Step by step:

1. **Embed** — each ID becomes its 16-dimensional row from the embedding table. Now the
   row is `(sequence_length, 16)`.
2. **Average** — collapse the sequence: take the mean across all the word vectors. Now
   it's a single `(16,)` vector, no matter how long or short the sentence was.
3. **Linear** — the neuron computes a weighted sum `z = w·v + b` of those 16 numbers.
4. **Sigmoid** — squashes `z` into `(0, 1)`.

### Why "average everything" is a legitimate trick

Think about what the average vector represents. If a review is mostly positive words, the
embedding of each positive word points somewhere in "positive" territory, and their
average lands in that same territory. If the review is negative, the average lands on the
other side. The linear neuron then just has to learn a boundary between the two regions.

This is called a **Bag of Words** model because the sentence is treated as an unordered
*bag* of words — the average throws away word order. "The movie was good" and "good was
the movie" average to the same thing. Order is often not that important for sentiment
(two words), so the model still works surprisingly well. The *big* limitation: "not
good" and "good" look almost identical in a bag. Fixing that is exactly why transformers
add positional information next problem.

### The sigmoid, concretely

The sigmoid function is:

```
σ(z) = 1 / (1 + e^(-z))
```

- `z = 0` → `σ = 0.5` (exactly uncertain).
- `z` large and positive → `σ` close to `1` (confident positive).
- `z` large and negative → `σ` close to `0` (confident negative).

It's the standard way to turn an unbounded number into a probability, because output is
always squeezed between 0 and 1.

### What about the padding zeros?

Short sentences are padded with ID `0`. Its embedding row is trained (or initialized)
to be near zero, so the padded words contribute almost nothing to the average. That's the
same design as problem 21: padding is free real estate that doesn't distort the signal.

## The implementation

```
import torch.nn as nn

class SentimentModel(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, 16)   # (V, 16) lookup table
        self.linear = nn.Linear(16, 1)                       # one output neuron
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x.long())      # (batch, seq, 16)
        averaged = embedded.mean(dim=1)          # (batch, 16)
        logit = self.linear(averaged)            # (batch, 1)
        return self.sigmoid(logit)               # (batch, 1), values in [0, 1]
```

Walkthrough:

1. **`super().__init__()`** — required boilerplate; it initializes the underlying
   `nn.Module` machinery so your layers register properly.
2. **`nn.Embedding(vocabulary_size, 16)`** — the lookup table from problem 20, sized to
   hold every token the vocabulary can produce.
3. **`nn.Linear(16, 1)`** — one neuron with 16 weights and a bias, outputting the single
   logit `z`.
4. **`nn.Sigmoid()`** — no weights, just the squashing function.
5. **`x.long()`** — embedding lookup requires integer indices; the grader may pass floats,
   so casting protects you.
6. **`mean(dim=1)`** — average over the *sequence* dimension (index 1). Averaging
   `dim=0` instead would average across sentences in the batch, which is a bug.
7. Output shape is `(batch, 1)` — a per-sentence probability — which is what the judge
   checks.

## Where you'll see this again

- **Positional encoding** (problem 23): the fix for the "bag" — gives the model order
  information so "not good" and "good" stop looking identical.
- **Self-attention** (problems 25-26): the smarter replacement for `mean(dim=1)` — a
  *weighted* average where the model decides which words matter.
- **Training loop / GPT**: this exact forward pass, scaled up, trained, and stacked into
  transformer blocks.

## Gotchas / common mistakes

- **Forgetting the sigmoid** — raw logits are unbounded and not a probability. The
  output must live in `[0, 1]`.
- **Averaging the wrong dimension** — `mean(dim=1)` collapses the sequence; `dim=0`
  collapses the batch and leaves you the wrong shape.
- **Not casting `x` to integers** — `nn.Embedding` rejects float indices.
- **Adding a training loop** — the spec explicitly asks for `__init__` and `forward`
  only.
- **Forgetting `super().__init__()`** — the layers won't register and the model breaks.

## One-line summary

Sentiment analysis is a Bag of Words pipeline — embed each word, average the embeddings
to throw away order, push the average through one neuron and a sigmoid — and that
average-then-classify template powers most of NLP.