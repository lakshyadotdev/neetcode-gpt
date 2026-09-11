# 21 — Intro to Natural Language Processing

## Why this problem exists

Last problem you looked up embeddings by token ID. But where do token IDs come from?
Before any neural network can look at text, someone has to decide *what counts as a
word* and give each word a number. That step is **tokenization**, and it's the very first
thing that happens in every NLP pipeline:

> text goes in → a sequence of integers comes out → those integers index an embedding table.

This problem builds a simple word-level tokenizer from scratch, which is the exact
machinery every language model starts with.

## Where this gets used

- Every NLP pipeline: sentiment analysis, translation, chatbots — all of them tokenize
  first.
- Real LLMs (GPT-4, LLaMA) use a fancier scheme called **Byte Pair Encoding (BPE)**,
  which splits rare words into smaller known fragments. Same idea as here, just a
  smarter vocabulary. You'll build BPE yourself later in the GPT section.
- **Padding** is used everywhere: any time you batch variable-length sentences, short
  ones get filled with zeros so every row has the same width.

## The setup

We get two lists of sentences, `positive` and `negative`, with `N` sentences each. We
must:

1. Collect every unique word across *both* lists.
2. Sort those words alphabetically (lexicographically) and assign IDs `1, 2, 3, ...`.
3. Convert each sentence into its sequence of IDs.
4. Pad short sentences with `0`s so all rows are the same length.
5. Stack into one tensor of shape `(2N, T)`, where `T` is the longest sentence's word
   count. Positive sentences first (in their original order), then negative.

The spec's example:

```
positive = ["Dogecoin to the moon"]
negative = ["I will short Tesla today"]

output = [[1.0, 7.0, 6.0, 4.0, 0.0],
          [2.0, 9.0, 5.0, 3.0, 8.0]]
```

## Main theory

### Splitting text into words

The simplest tokenizer is just `sentence.split()` — split on spaces. "Dogecoin to the
moon" becomes `["Dogecoin", "to", "the", "moon"]`. Four words. That's the whole
tokenization step for this problem (real tokenizers also worry about punctuation,
lowercasing, and sub-words — we don't need any of that here).

### Building the vocabulary

The **vocabulary** is the set of unique words we can handle. Collect every word from
every sentence, keep only the distinct ones:

```
Dogecoin, I, Tesla, moon, short, the, to, today, will
```

Sort them alphabetically and number them starting at **1**:

```
Dogecoin -> 1      moon  -> 4      to    -> 7
I       -> 2      short -> 5      today -> 8
Tesla   -> 3      the   -> 6      will  -> 9
```

(Sorting is what makes this *deterministic*: run it twice, same IDs every time.)

Now convert each sentence:

- "Dogecoin to the moon" → `[1, 7, 6, 4]` (4 words)
- "I will short Tesla today" → `[2, 9, 5, 3, 8]` (5 words)

### Padding: why the zeros?

The second sentence has 5 words, the first has 4. Matrix operations require every row to
be the same width, so we can't just stack them. We pad the short one with `0`s until it
matches the longest sentence: `[1, 7, 6, 4, 0]`. The maximum length `T` is the word count
of the longest sentence.

Notice the trick: token IDs start at `1`, so `0` is free to mean "nothing here." The
embedding at row 0 can just be a vector of zeros, and it contributes nothing to
downstream averages.

### Why sub-word tokenizers exist (preview)

A word-level vocabulary breaks on anything new: misspellings, names, made-up words.
BPE fixes this by keeping a vocabulary of *fragments* ("low", "est", "ing", ...) and
splitting unknown words into fragments it does know. Same integer-ID machinery, smaller
vocabulary, no word left behind. That's a whole later section — just know this problem
is the foundation.

## The implementation

```
import torch

def encode_sentences(positive, negative):
    # 1. Collect every unique word across both lists
    vocab = set()
    for sentence in positive + negative:
        for word in sentence.split():
            vocab.add(word)

    # 2. Sort, assign IDs 1, 2, 3, ...
    word_to_id = {word: i + 1 for i, word in enumerate(sorted(vocab))}

    # 3. Turn a sentence into a tensor of token IDs
    def encode(sentence):
        return torch.tensor([word_to_id[w] for w in sentence.split()])

    # 4. Encode all sentences, positives first
    sequences = [encode(s) for s in positive] + [encode(s) for s in negative]

    # 5. Pad to a rectangle and stack
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
```

Walkthrough:

1. **`vocab = set()`** — a set naturally holds only unique words. We iterate both lists
   combined (`positive + negative`), so the vocabulary covers all the text.
2. **`sorted(vocab)`** — alphabetical order; `enumerate` gives `0, 1, 2, ...`, so `i + 1`
   shifts the IDs to start at 1, leaving `0` for padding.
3. **`encode`** — splits a sentence and maps each word through the dictionary, returning
   a 1D tensor of IDs.
4. **List order matters** — encoding all positives, then all negatives, gives exactly the
   `(2N, T)` layout the spec wants.
5. **`pad_sequence(..., batch_first=True, padding_value=0)`** — stacks variable-length
   tensors into one rectangle, filling gaps with `0`. `batch_first=True` makes the batch
   dimension first, so rows are sentences.

## Where you'll see this again

- **Sentiment analysis** (problem 22): feeds the output of this tokenizer into an
  embedding layer. The full pipeline is tokenize → embed → classify.
- **Word embeddings** (problem 20): the IDs produced here are the row numbers looked up
  in the embedding table.
- **Tokenization edge cases / build-vocabulary** (GPT section): the same ideas, scaled up
  to sub-word BPE tokens and trained from gigabytes of text.

## Gotchas / common mistakes

- **Forgetting `sorted`** — the spec demands lexicographic IDs. A set's iteration order
  is arbitrary, so never iterate it directly.
- **Starting IDs at 0** — padding uses `0`, so real tokens must start at `1`.
- **Padding with the wrong value** — must be `0`, not the padding token's own ID.
- **Negatives first** — output rows must be all positives, then all negatives.
- **Padding by hand** — easy to get wrong; `pad_sequence` does the right thing if you
  pass `batch_first=True` and `padding_value=0`.

## One-line summary

Tokenization splits text into words, sorts them alphabetically, numbers them from 1, and
pads every sentence with zeros to the longest length — turning arbitrary text into the
rectangular integer tensors a neural network can actually eat.