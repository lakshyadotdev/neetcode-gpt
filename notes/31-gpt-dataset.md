# 31 — GPT Dataset

## Why this problem exists

The data loader (problem 30) worked with pre-tokenized integers. Real life doesn't
hand you that — you have raw text, often gigabytes of it. When people say "ChatGPT was
trained on the internet," this step makes it concrete: a raw text corpus gets sliced
into overlapping `(context, next_token)` training pairs, ready for the model.

This problem takes the same shift-by-one idea from problem 30 and pushes it one step
earlier in the pipeline: start from the string itself.

## Where this gets used

- This is the dataset-preparation step in training any autoregressive model from raw
  text (Karpathy's GPT videos, every LLM training pipeline).
- It shows why even a short document yields *many* training examples — a 1,000-word
  text with `context_length = 128` produces ~872 overlapping examples.
- The objective — "given the tokens so far, predict what comes next" — is the single
  objective of GPT, LLaMA, Claude, and friends.

## The setup

```
raw_dataset = "Hello darkness my old friend"
context_length = 3
batch_size = 2

Output:
X = [['darkness', 'my', 'old'],
     ['hello', 'darkness', 'my']]
Y = [['my', 'old', 'friend'],
     ['darkness', 'my', 'old']]
```

- `raw_dataset` — a string of text, split on whitespace into tokens.
- `context_length` — number of consecutive tokens per training sequence (≥ 1).
- `batch_size` — independent sequences to sample per call (≥ 1).

Write a `batch_loader()` that samples `batch_size` random start indices with
`torch.randint()`, slices the tokenized text into (input, label) pairs of length
`context_length`, and returns `(X, Y)` with identical shapes, Y shifted one position
right of X.

## Main theory

### The next-token prediction objective

Every autoregressive language model learns one thing: given the tokens so far, predict
what comes next. "The cat sat on the" → "mat". Scale that to trillions of tokens of
internet text and you get a system that writes code and solves math. The whole dataset
problem is just: manufacture as many of these little (context → next) examples as
possible from raw text.

### The sliding window

For a document of length `L` and context length `C`, any position `i` in
`[0, L - C - 1]` gives one training example:

```
X_i = [token_i, ..., token_{i+C-1}]
Y_i = [token_{i+1}, ..., token_{i+C}]
```

Same window, shifted one token. Because we slide the window over *every* position, the
examples overlap heavily — a single Wikipedia article produces thousands of them. That
overlap is fine and expected: each example is a slightly different view of the same
text.

### The context window (why C matters)

`C` is the **context window** — how many previous tokens the model may look at. GPT-2
used 1,024 tokens; modern models like GPT-4 support 128K or more. A bigger context
means the model can reference earlier paragraphs, but standard attention's compute and
memory cost grow quadratically with C, so it's a real trade-off.

### The example, concretely

Split "Hello darkness my old friend" on whitespace and lowercase (the expected output
shows lowercase words): `["hello", "darkness", "my", "old", "friend"]`, length 5.

The sampler drew start index 1 first, then 0:

- Start 1: `X = ['darkness', 'my', 'old']`, `Y = ['my', 'old', 'friend']` — "darkness"
  predicts "my", "darkness my" predicts "old", "darkness my old" predicts "friend".
- Start 0: `X = ['hello', 'darkness', 'my']`, `Y = ['darkness', 'my', 'old']` — same
  logic from the start of the sentence.

Two of the three possible windows, sampled at random.

## The implementation

```python
import torch

def batch_loader(raw_dataset: str, context_length: int, batch_size: int):
    tokens = [w.lower() for w in raw_dataset.split()]   # words → lowercase tokens
    ix = torch.randint(0, len(tokens) - context_length, (batch_size,))

    X = [tokens[i:i + context_length] for i in ix]
    Y = [tokens[i + 1:i + 1 + context_length] for i in ix]
    return X, Y
```

Walkthrough:

1. `raw_dataset.split()` splits on whitespace: `["Hello", "darkness", "my", "old",
   "friend"]`. `.lower()` makes each word lowercase to match the expected output — in
   a real pipeline you'd instead look each word up in the vocabulary from problem 28
   and get integers (that's what the NLP Intro problem's encoder does).
2. `torch.randint(0, len(tokens) - context_length, (batch_size,))` — `batch_size`
   random starts, bounded so a full window always fits: the last valid start is
   `L - C`, giving a window ending exactly on the last token (index `L - 1`).
3. `X` slices `context_length` tokens from each start: `tokens[i : i + C]`.
4. `Y` slices `tokens[i + 1 : i + 1 + C]` — the identical window shifted one right.
   That one shifted slice *is* the label: every position of X is paired with the token
   that follows it.
5. We return plain lists of word lists here so the shift is readable. In the real
   pipeline (and in problem 30) the same shapes carry integers instead.

## Where you'll see this again

- **Problem 30's data loader** is the integer version of this exact slicing.
- **Training-loop problems**: this `(X, Y)` goes straight into the model and loss.
- Any LLM training pipeline from raw text, including Karpathy's "Let's build GPT".

## Gotchas / common mistakes

- **Shifting Y on only one side.** Y must start at `i + 1` and end at
  `i + 1 + context_length`, keeping the same length as X.
- **Off-by-one on valid starts.** The last usable start is `L - C`, not `L - C + 1`,
  or the window overshoots the document.
- **Forgetting to lowercase (or encode).** The expected output is lowercase words; a
  real pipeline would encode words to ints with the vocabulary before training.
- **Overlapping examples are correct.** Don't try to make windows non-overlapping —
  the sliding window is supposed to produce overlapping examples.
- **Matching shapes.** X and Y must both be `(batch_size, context_length)`.

## One-line summary

A GPT dataset = split raw text into tokens, sample random start points, and for each
one slice X and the same window shifted one token right as Y — turning any document
into thousands of "predict the next token" training examples.