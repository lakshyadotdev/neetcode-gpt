# 27 — Tokenizer (Byte Pair Encoding)

## Why this problem exists

A model can't read text — it reads numbers. Before anything else, text has to be
chopped up into **tokens**, and each token given an ID. In the NLP Intro problem we did
the naive thing: split on spaces, one word = one token. That works until you meet a
word the model has never seen: "unforgettable", `getElementById`, a typo. There's no
token for it, so it becomes garbage (or an "unknown" placeholder).

You could go the other extreme and split into single characters — then every word is
representable. But "c", "a", "t" doesn't really convey "cat", and the model has to
relearn meaning from scratch at the character level.

**Byte Pair Encoding (BPE)** is the middle ground almost every modern LLM uses:

- common words stay whole ("the", "and", "cat"),
- rare words decompose into pieces that still carry meaning ("un" + "forgett" + "able").

## Where this gets used

- GPT-2, GPT-3, GPT-4 (about 100,000 tokens), LLaMA — basically every modern LLM
  tokenizes with BPE.
- This is step one of the GPT pipeline we're building: Tokenize → Data → Model →
  Train → Generate. Everything after this assumes tokens exist.
- You can watch any text get tokenized live at tiktokenizer.vercel.app.

## The setup

Inputs:

- `corpus` — a string of text
- `num_merges` — how many merge operations to run (`> 0`)

Output: the list of merges performed, each one a `[token_a, token_b]` pair.

```
corpus = "abcabc"
num_merges = 2

Output: [["a","b"], ["ab","c"]]
```

This is the BPE *training* algorithm — we learn the merges from data. (Encoding a new
string with the learned merges is a separate step that real tokenizers do afterwards,
and it shows up again in problem 29.)

## Main theory

### The vocabulary problem, in one line

Word-level splits lose rare words; character-level splits lose meaning. BPE learns
which pieces deserve to be their own token, straight from the training data.

### The BPE training loop

1. Split the corpus into individual characters.
2. Count how often every adjacent pair of tokens appears.
3. Pick the most frequent pair. On a tie, choose the **lexicographically smallest**
   pair (i.e. the one that comes first alphabetically, comparing the strings).
4. Replace all *non-overlapping* occurrences of that pair, scanning left to right,
   gluing the two tokens into one new token.
5. Repeat until you've done `num_merges` merges.

### A tiny worked example

`corpus = "abcabc"`:

- Step 1: characters → `['a','b','c','a','b','c']`
- Step 2: count adjacent pairs:
  - `('a','b')` → 2
  - `('b','c')` → 2
  - `('c','a')` → 1
- Step 3: tie between `('a','b')` and `('b','c')` at 2 each. Lexicographically
  `('a','b') < ('b','c')` ('a' comes before 'b'), so we merge 'a' + 'b'.
- Step 4: `['a','b','c','a','b','c']` → `['ab','c','ab','c']`
- Round 2: pairs are now `('ab','c')` → 2 and `('c','ab')` → 1. Merge 'ab' + 'c':
  `['abc','abc']`.

Two merges done → `[["a","b"], ["ab","c"]]`.

"Non-overlapping, left to right" matters: if the sequence were `'a','b','a','b'`,
merging `('a','b')` gives `['ab','ab']`, not `['ab','a','b']` — each token is used in
at most one merge.

### How BPE encoding works afterwards (for context)

After training, encoding a new string means applying the learned merges in the order
they were learned: first merge everywhere it appears, then the second, and so on. That
deterministically converts any text into subword tokens. GPT-2 performs 50,000 merges
during training; GPT-4 ends up with around 100,000 tokens in its vocabulary.

## The implementation

```python
def get_merges(corpus: str, num_merges: int) -> list[list[str]]:
    tokens = list(corpus)          # step 1: split into characters
    merges = []

    for _ in range(num_merges):
        # step 2: count adjacent pairs
        pairs = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] = pairs.get(pair, 0) + 1

        if not pairs:              # no pairs left to merge
            break

        # step 3: most frequent pair, ties → lexicographically smallest
        best = min(pairs, key=lambda p: (-pairs[p], p))
        merges.append(list(best))

        # step 4: replace all non-overlapping occurrences, left to right
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best:
                new_tokens.append(tokens[i] + tokens[i + 1])
                i += 2             # consume both tokens
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    return merges
```

Walkthrough:

1. `tokens = list(corpus)` — "abcabc" becomes `['a','b','c','a','b','c']`. Each
   character is its own token, exactly like step 1 says.
2. The pair-counting loop walks the *current* token list and counts each adjacent pair
   in a dict. A tuple `(tokens[i], tokens[i+1])` is the key so the pair survives as a
   unit later.
3. `best = min(pairs, key=lambda p: (-pairs[p], p))` — the whole "most frequent,
   ties break lexicographically" rule in one line. Sorting by `-pairs[p]` puts the
   most frequent pair first (negating flips largest to smallest), and `p` as the
   tiebreaker picks the alphabetically smaller pair.
4. The merge pass rebuilds `new_tokens`. When the two current tokens match `best`, we
   glue them into one string (`tokens[i] + tokens[i+1]`) and jump `i` by 2 so a token
   can never be merged twice (no overlapping merges). Otherwise keep the token and
   advance by 1.
5. `if not pairs: break` — if the corpus was short and everything already merged into
   one token, there's nothing left to merge even though `num_merges` isn't done.
6. Each merge is recorded in order, so the returned list is the exact merge history.

## Where you'll see this again

- **Problem 28 (Build Vocabulary)**: the simpler character-level vocab your GPT
  project actually uses, to keep training feasible on a laptop.
- **Problem 29 (Tokenization Edge Cases)**: the weird consequences of a learned
  vocabulary — why GPT can't do arithmetic, and why some languages cost more tokens
  per word.
- **Problems 30–31**: the data loader and dataset feed the integer IDs produced by
  tokenization into the model.

## Gotchas / common mistakes

- **Forgetting the tie-break.** When two pairs tie on frequency, pick the
  lexicographically smallest — it's what makes the result deterministic and matches
  the expected output.
- **Merging overlapping pairs.** Scan left to right and advance by 2 after a merge, or
  `'a','b','a','b'` would merge into `['ab','a','b']` instead of `['ab','ab']`.
- **Not recounting pairs after each merge.** Counts must come from the *current* token
  list, not the original characters — the merged token changes the pair landscape
  (that's the whole point).
- **Running out of pairs.** If `num_merges` exceeds what the corpus allows, stop
  gracefully instead of crashing.
- **Trying to encode instead of train.** This problem asks for the merge history
  (`[["a","b"],["ab","c"]]`), not the tokenized output string.

## One-line summary

BPE training = repeatedly find the most frequent adjacent token pair, glue it into one
token, and repeat — building a vocabulary of whole common words plus meaningful pieces
of rare ones.