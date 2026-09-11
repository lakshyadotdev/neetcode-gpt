# 03 — Softmax

## Why this problem exists

A model's last layer spits out raw **scores** for each possible answer — one per class,
or (for GPT) one per possible next word. Those scores can be anything: negative, huge,
whatever. They aren't probabilities yet.

Softmax is the machine that turns a vector of raw scores into a **probability
distribution**: every output is positive, and they all add up to exactly 1, so you can
read them as "the model thinks class A has this much chance".

Without this step there's no clean way to go from "a big number" to "a confidence".

## Where this gets used

- **Classification**: the last layer outputs one score per class, and softmax converts
  them into probabilities.
- **GPT**: the final layer scores every possible next token, and softmax turns those into
  probabilities — "this token has a 70% chance, that one 20%..." This is literally how the
  model decides which word comes next.
- **Attention** (later in the course): softmax turns attention scores into weights that
  sum to 1, deciding how much each token "looks at" every other token.

## The setup

NeetCode gives you a 1D NumPy array `z` of logits. Round the result to **4 decimals**.

Example: `softmax([1, 2, 3]) → [0.09, 0.2447, 0.6652]`. Equal inputs give equal outputs:
`softmax([1, 1, 1]) → [0.3333, 0.3333, 0.3333]`.

## Main theory

The formula for the `i`-th element:

```
softmax(z)_i = e^(z_i) / sum of e^(z_j)  for all j
```

That is: raise every score to the power of `e`, add them all up, and divide each by the
total. Three things are happening:

1. **`e^x` makes everything positive.** No matter how negative a score is, `e^x` is
   always positive — so we never get stuck with negative "probabilities".
2. **Bigger scores get amplified.** Since `e^x` grows fast, the biggest score grabs a
   lopsided share. `[1, 2, 3]` does *not* become `[0.16, 0.33, 0.5]` (that would be plain
   dividing by the sum). It becomes `[0.09, 0.24, 0.66]` — the winner is exaggerated.
   That's why softmax is sometimes called "**soft** argmax": like picking the winner
   (argmax), but softly, giving the top score the most probability instead of all of it.
3. **Dividing by the sum** makes everything total exactly 1.

Let's verify the example by hand. `e^1 = 2.718`, `e^2 = 7.389`, `e^3 = 20.086`. Their sum
is `30.193`. So:

- `softmax(1) = 2.718 / 30.193 = 0.09`
- `softmax(2) = 7.389 / 30.193 = 0.2447`
- `softmax(3) = 20.086 / 30.193 = 0.6652`

The biggest input got the biggest share, and the three add up to `0.09 + 0.2447 + 0.6652 ≈ 1`.

### The critical trick: subtract the max

For numerical stability, the safe version is:

```
softmax(z)_i = e^(z_i - max(z)) / sum of e^(z_j - max(z))
```

**Why?** If a logit is `1000`, then `e^1000` is infinity in a computer — it **overflows**
to `nan` and your whole answer is garbage. But if you subtract the max first, the largest
exponent becomes `e^(1000 - 1000) = e^0 = 1`, which is always safe.

**Does it change the answer?** No. You're dividing top and bottom by the same `e^max(z)`,
so it cancels out. Mathematically identical, numerically safe. Always do this in practice.

## The implementation

```python
import numpy as np

def softmax(z):
    z = z - np.max(z)              # subtract the max for stability
    exp_z = np.exp(z)              # element-wise e^z
    return np.round(exp_z / np.sum(exp_z), 4)
```

Walkthrough:

1. `z = z - np.max(z)` — shift all logits down by the biggest one, so the largest exponent
   is `e^0 = 1`. Prevents overflow.
2. `np.exp(z)` — element-wise `e^z` for the whole array in one shot.
3. `np.sum(exp_z)` — the denominator, the total of all the `e^z` values.
4. `exp_z / np.sum(exp_z)` — element-wise division, so each value becomes its fraction of
   the total. Everything now sums to 1.
5. `np.round(..., 4)` — NeetCode expects 4 decimal places.

## Where you'll see this again

- **Cross-entropy loss** (problem 04) pairs with softmax — softmax makes the
  probabilities, cross-entropy measures how wrong they are. Their combined gradient works
  out to a clean `predicted - actual`, which is why they're always used together.
- **Transformers / GPT**: softmax shows up twice — at the output to pick the next token,
  and inside the attention mechanism to weight how tokens look at each other.

## Gotchas / common mistakes

- **Forgetting to subtract the max** — with large logits you get `inf`/`nan`.
- **Using `max` (one number) where you need the array** — `np.max(z)` is the scalar max;
  keep `z` as an array for the element-wise subtraction.
- **Not normalizing by the sum** — each element must be divided by the *total* so they sum
  to 1.
- **Rounding too early** — compute everything, then round the final array to 4 decimals.
- **Plain `max(z, 0)`-style confusion** — softmax is not a per-element threshold; the
  denominator uses all elements together.

## One-line summary

Softmax turns a vector of raw scores into probabilities that are positive and sum to 1,
amplifying the winner — and you always subtract the max first so huge logits don't
overflow.