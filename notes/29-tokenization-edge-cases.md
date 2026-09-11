# 29 — Tokenization Edge Cases

## Why this problem exists

Your character-level vocabulary works, but real tokenizers (BPE with ~100k learned
tokens) produce genuinely surprising behavior — and that behavior explains real LLM
quirks.

Here's the famous one: **why can't GPT do 42 + 37?** Because "42" might be a single
token in its vocabulary, while "142" gets split into `["1", "42"]`. The model never
sees digits in consistent positions, so arithmetic patterns are nearly impossible to
learn from token sequences alone. That's not a model failure — it's a direct
consequence of how the vocabulary was built from data.

## Where this gets used

- Explains why LLMs are bad at arithmetic, string reversal, and similar tasks.
- Explains token cost: a language that splits into more pieces per word costs more
  tokens — more money, and less content fits in the context window.
- Helps you read tokenizer output (tiktokenizer.vercel.app) and reason about prompts.

## The setup

Implement three functions, all built on one algorithm: **greedy left-to-right longest
match** tokenization.

The algorithm, precisely:

- Start at the left of the text.
- Find the *longest* substring starting here that exists in the vocabulary.
- Consume that substring as a token, move past it, and repeat.
- If no match exists at all, consume a single character.

```
numbers = [12, 34]
vocab = {"0":0, "1":1, "2":2, "3":3, "4":4}

tokenize_numbers = [["1","2"], ["3","4"]]
# single-digit vocab → every digit its own token, perfectly consistent

numbers = [2249, 2250]
vocab = {"0":10, "1":11, "2":12, "22":16, "225":18, "49":19}

tokenize_numbers = [["22","49"], ["225","0"]]
# 2249 splits at "22"; 2250 splits at "225" — 1 apart, totally different shapes!
```

## Main theory

### Why consecutive numbers split differently

The vocabulary is learned from data, not designed. Whatever strings appeared often in
training became tokens. So:

- "380" might be one token, "381" might be `["3","81"]`, and "382" might be `["38","2"]`.

The model sees completely different structures for numbers one apart. There's no
consistent "one digit at a time" signal to learn arithmetic from — each number is
almost a different little language.

### The greedy algorithm, worked

Tokenizing "2249" with the vocab above:

1. At position 0, try the longest match: "2249" ✗, "224" ✗, "22" ✓ → token "22".
2. From position 2: "49" ✓ → token "49".

Result: `["22", "49"]`.

Tokenizing "2250":

1. "2250" ✗, "225" ✓ → token "225".
2. From position 3: "0" ✓ → token "0".

Result: `["225", "0"]`.

Same digits, totally different grouping. That's the inconsistency in miniature.

### Count and fertility

- `count_tokens(text, vocab)` — run greedy tokenization and count the tokens. Text
  that matches multi-character vocabulary entries uses fewer tokens.
- **Fertility** = tokens-per-word. Split text by spaces to count words, then
  `fertility = token_count / word_count`. A score of 1.0 means one token per word
  (ideal). English in GPT-4 averages ~1.3. Some languages score 3–6× higher: the same
  meaning costs many more tokens, the model has seen less training data per token, and
  less content fits in the context window.

### Why greedy is OK but not perfect

Left-to-right longest match is fast and deterministic, but it doesn't always find the
globally fewest tokens. Real BPE *encoding* doesn't use greedy matching at all — it
replays the merges learned during training, in order. Greedy is a stand-in that
captures the same flavor of behavior for this problem.

## The implementation

```python
def tokenize(text: str, vocab: dict[str, int]) -> list[str]:
    tokens = []
    i = 0
    while i < len(text):
        matched = False
        # longest match first: try the whole remaining string, then shrink
        for j in range(len(text), i, -1):
            if text[i:j] in vocab:
                tokens.append(text[i:j])
                i = j
                matched = True
                break
        if not matched:            # nothing in vocab → consume one character
            tokens.append(text[i])
            i += 1
    return tokens

def tokenize_numbers(numbers: list[int], vocab: dict[str, int]) -> list[list[str]]:
    return [tokenize(str(n), vocab) for n in numbers]

def count_tokens(text: str, vocab: dict[str, int]) -> int:
    return len(tokenize(text, vocab))

def fertility_score(text: str, vocab: dict[str, int]) -> float:
    words = len(text.split())
    return round(count_tokens(text, vocab) / words, 4)
```

Walkthrough:

1. The helper `tokenize` is the greedy loop. `for j in range(len(text), i, -1)` starts
   with the *longest* possible slice (`text[i:]`) and shrinks, so the first hit is
   always the longest substring in the vocab — that's the "longest match" part.
2. When nothing matches, we still consume one character. This keeps progress even if
   the text contains characters outside the vocabulary.
3. `tokenize_numbers` converts each number to a string first (`str(n)`) — vocab keys
   are strings, so 12 must become "12" before lookup. It returns one token list per
   number.
4. `count_tokens` is just the length of the token list — longer vocabulary matches
   mean fewer tokens.
5. `fertility_score` splits text on whitespace to count words, divides, and rounds to
   4 decimal places as the spec requires.

## Where you'll see this again

- Any real tokenizer: GPT's `cl100k_base` has ~100k tokens learned from data — same
  inconsistency, same fertility effects, bigger scale.
- Whenever you wonder why a model fails at math or why some languages cost more.
- The merge-order encoding mentioned here connects back to problem 27's training.

## Gotchas / common mistakes

- **Longest match means longest first.** Iterate the slice length from largest to
  smallest. Starting with single characters and growing gives the *shortest* match and
  wrong answers.
- **Numbers are ints, vocab keys are strings.** Convert with `str(n)` or lookups
  always miss.
- **"No match" still consumes one character.** Don't crash or loop forever on text
  not in the vocab.
- **Fertility rounds to 4 decimals** and divides by the *word* count, not the
  character count.
- **One token list per number.** `tokenize_numbers` returns a list of lists, not one
  flat list.

## One-line summary

A data-learned vocabulary means greedy tokenization is inconsistent — consecutive
numbers split differently and languages vary wildly in tokens per word — and that's
why token-trained models can't do arithmetic and some languages cost more.