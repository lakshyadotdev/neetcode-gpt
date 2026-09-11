# 28 — Build Vocabulary

## Why this problem exists

Neural networks only understand numbers. "Hello" means nothing to a tensor — you need
`[1, 0, 2, 2, 3]` before any math can happen. So the very first step of a language
model is a **vocabulary**: a fixed list of the symbols the model knows, each mapped to
an integer ID.

BPE (problem 27) is the fancy production version. This problem builds the simpler
character-level version: every unique character gets its own integer. And that's not a
toy — it's what your GPT project will actually use, because a character vocabulary is
tiny (dozens of entries), which keeps training feasible on a laptop. Karpathy's
nanoGPT does exactly this.

## Where this gets used

- The encode/decode pattern here is identical in every tokenizer, BPE or not: encode
  text → integers for the model, decode integers → text for humans.
- The output of `encode` is literally the input to the data loader (problems 30–31)
  and then to the model.
- At generation time, the model's output (IDs) is turned back into readable text
  through the same vocabulary.

## The setup

Build three functions:

1. `build_vocab(text)` — collect the unique characters, sort them, assign each an ID
   starting from 0. Return both mappings.
2. `encode(text, stoi)` — string → list of integers.
3. `decode(ids, itos)` — list of integers → string.

Example:

```
text = "hello"

stoi = {"e": 0, "h": 1, "l": 2, "o": 3}
itos = {0: "e", 1: "h", 2: "l", 3: "o"}

encode("hello", stoi)          = [1, 0, 2, 2, 3]
decode([1, 0, 2, 2, 3], itos)  = "hello"
```

## Main theory

### Two dictionaries, one idea

- **stoi** (string → int): used when encoding. Look up each character, get its ID.
- **itos** (int → string): used when decoding. Look up each ID, get its character.

They're exact inverses of each other, and the names are just short for "string to int"
and "int to string".

### Why sort?

`"hello"` has unique characters `{'e','h','l','o'}`. Sorted alphabetically that's
`['e','h','l','o']`, and enumerating gives them IDs 0, 1, 2, 3. Sorting makes the
mapping **deterministic**: the same text always produces the same vocabulary, which
means the same IDs, which means reproducible training. The actual numbers are
arbitrary — nothing in the model cares whether 'a' is 0 or 97 — but they must be
*consistent*.

### Why IDs from 0?

Later, the model looks up each token's ID in an **embedding table** — a matrix where
row `i` holds the learned vector for token `i`. Rows are indexed by position, so IDs
need to be small, contiguous integers starting at 0. (It's also why a small vocab keeps
the whole model small enough for a laptop.)

### Encode and decode are just lookups

- Encode: for each character, find its ID → a list of ints. That's the model's input.
- Decode: for each ID, find its character, then join them into a string. That's how
  model output becomes text again.

## The implementation

```python
def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set(text))            # unique chars, alphabetically sorted
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode(text: str, stoi: dict[str, int]) -> list[int]:
    return [stoi[ch] for ch in text]

def decode(ids: list[int], itos: dict[int, str]) -> str:
    return "".join(itos[i] for i in ids)
```

Walkthrough:

1. `set(text)` removes duplicates ("hello" → `{'e','h','l','o'}`); `sorted(...)` puts
   them in alphabetical order, so the same text always yields the same ordered list.
2. `enumerate(chars)` pairs each char with its position — 0, 1, 2, ... — and `stoi` is
   built from that. This is where sorting pays off: position becomes the ID.
3. `itos` is built by flipping `stoi`. Iterating `stoi.items()` gives `(char, id)`
   pairs, so `{i: ch for ch, i in ...}` produces `{id: char}`. Flipping guarantees the
   two dicts stay exact inverses.
4. `encode` is a single list comprehension: one dictionary lookup per character.
   `"hello"` → `[1, 0, 2, 2, 3]` (h→1, e→0, l→2, l→2, o→3).
5. `decode` looks up each ID and joins the results back into one string. `itos[2]` is
   `"l"`, so both 2s decode to 'l'. The round-trip holds: `decode(encode(text))`
   gives back the original text.

The spec's second example works identically: `"abcabc"` has unique sorted chars
`['a','b','c']`, so `stoi = {"a":0, "b":1, "c":2}`, and
`encode("cab") = [2, 0, 1]`, `decode([2, 0, 1]) = "cab"`.

## Where you'll see this again

- **Problem 29**: tokenization edge cases — what greedy tokenization with a learned
  vocabulary does to numbers and to different languages.
- **Problems 30–31**: the data loader and dataset slice the *integer* sequences this
  function produces.
- **Training and generation**: the model outputs a probability distribution over vocab
  entries; you pick an ID and `decode` it to read the answer.
- Modern LLMs run the same pattern with ~100k BPE tokens instead of ~100 characters.

## Gotchas / common mistakes

- **The two dicts must be inverses.** If you build `itos` independently you risk them
  drifting apart; flipping `stoi` guarantees they always match.
- **Forgetting to sort.** Without sorting, the same text could produce a different
  mapping depending on iteration order — non-deterministic.
- **`decode` needs `join`, not a list.** Returning a list of characters isn't text.
- **IDs must start at 0 and be contiguous.** They'll index an embedding table later;
  gaps or a non-zero start break that.
- **Case sensitivity.** 'H' and 'h' are different characters and get different IDs —
  normal, but worth remembering when you see different vocab sizes.

## One-line summary

Build a vocabulary = collect the unique sorted characters, assign each an integer ID,
and hand the model an encode/decode pair so text becomes integers and integers become
text.