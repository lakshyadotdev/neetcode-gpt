# 04 — Cross-Entropy Loss

## Why this problem exists

Softmax gave you probabilities — the model's confidence in each class. But confidence
alone doesn't tell the model *how wrong* it was. If it predicted "cat" at 99% and the
answer was "dog", that should hurt a lot; if it predicted "cat" at 60% and the answer was
"cat", that's fine.

**Cross-entropy loss** turns a predicted distribution plus the true answer into a single
number measuring how wrong the model was: low when it's right, high when it's confidently
wrong. This is *the* number gradient descent tries to shrink. It's the loss behind almost
every classifier and behind every token prediction in GPT.

## Where this gets used

- **Classification**: the standard loss for any model that outputs class probabilities.
- **GPT**: every next-token prediction is a softmax over the vocabulary, and cross-entropy
  is how the model is punished for predicting the wrong word.
- It's the **loss function** — the thing the whole gradient-descent loop (problem 01) is
  minimizing. Without a good loss, nothing to learn from.

## The setup

NeetCode gives two versions. Round the result to **4 decimals**.

**Binary** (two classes), where `y_i` is the true label (0 or 1) and `p_i` is the
predicted probability:

```
L = -(1/n) * sum over i of [ y_i * ln(p_i) + (1 - y_i) * ln(1 - p_i) ]
```

**Categorical** (many classes), where `y_(i,c)` is 1 if sample `i` is class `c` (one-hot)
and `p_(i,c)` is the predicted probability:

```
L = -(1/n) * sum over i, sum over c of [ y_(i,c) * ln(p_(i,c)) ]
```

## Main theory

### The core idea: "how surprised were you?"

Only one number in the prediction really matters: **the probability the model assigned to
the correct answer**. Cross-entropy is basically `-ln(probability of the right answer)`,
averaged over all samples.

- Correct class got probability `0.99` → loss `-ln(0.99) ≈ 0.01` (barely surprised).
- Correct class got probability `0.01` → loss `-ln(0.01) ≈ 4.6` (very surprised).

### The `y` is just a switch

The true label isn't part of the math — it *selects* which term to keep. In binary, if
`y = 1` the second term (`(1 - y) * ...`) dies, and if `y = 0` the first term dies. So you
always end up with the log of the probability assigned to the truth. In categorical, the
one-hot `y` does the same: it zeroes out every class except the correct one.

### Why the log punishes so hard

The log is what makes this more than "average the probabilities":

- confident + correct (`p = 0.99`) → tiny penalty, `-ln(0.99) ≈ 0.01`
- unsure (`p = 0.5`) → moderate penalty, `-ln(0.5) ≈ 0.69`
- confident + wrong (`p = 0.01`) → huge penalty, `-ln(0.01) ≈ 4.6`

Because `ln` crashes toward `-infinity` as its input approaches 0, the loss brutally punishes
a model that's confidently wrong. That forces the model to be well-calibrated — you can't
win by shouting wrong answers at 99%.

### The `ln(0)` gotcha

If the model predicts exactly `0` for the correct class, `ln(0)` is undefined (`-inf`). So
in practice you **clip** predictions into `[epsilon, 1 - epsilon]` with a tiny epsilon like
`1e-7` to avoid ever taking `ln(0)`.

## The implementation

```python
import numpy as np

def cross_entropy(y_true, y_pred):
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)   # avoid ln(0)
    # binary: y is 0/1; categorical: y is one-hot
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    return np.round(loss, 4)
```

Walkthrough:

1. `np.clip(y_pred, eps, 1 - eps)` — clamps predictions away from exactly 0 or 1, so
   `ln` never blows up. Matches the standard practice.
2. `y_true * np.log(y_pred)` — for each sample, the one-hot/binary `y` zeroes out every
   term except the correct class, leaving the log of the correct probability.
3. `np.sum(..., axis=1)` — sums across classes for each sample (for binary, the 0/1
   switch already left just one surviving term per sample).
4. `np.mean(...)` — the `1/n` average over all samples.
5. The leading `-` flips the logs (which are negative) to a positive loss.
6. `np.round(..., 4)` — NeetCode wants 4 decimals.

**Check against the binary example.** `y_true = [1, 0, 1]`, `y_pred = [0.9, 0.1, 0.8]`:

- sample 1 (`y = 1`): keep `ln(0.9)`
- sample 2 (`y = 0`): keep `ln(1 - 0.1) = ln(0.9)`
- sample 3 (`y = 1`): keep `ln(0.8)`

Average: `-(1/3)[ln(0.9) + ln(0.9) + ln(0.8)] = 0.1446`. Confident and correct, so the
loss is low.

## Where you'll see this again

- **Handwritten-digit classifier** (problem 14) and **sentiment analysis** (problem 18)
  both end in softmax + cross-entropy.
- In **backpropagation** (problems 08-09) you'll see the happy fact that the gradient of
  softmax+cross-entropy together is simply `predicted - actual` — which is why they're
  always paired.
- **Training GPT**: the model's loss at every step is cross-entropy over the next token.

## Gotchas / common mistakes

- **Not clipping predictions** — a 0 in `ln` gives `-inf` / `nan`.
- **Using natural vs any log** — it's `ln` (natural log); with probabilities it works out
  consistently, but keep it matching the spec.
- **Forgetting the minus sign** — logs of probabilities are negative, so you need the `-`
  to get a positive loss.
- **Forgetting to average** over samples (`1/n`).
- **Averaging over the wrong axis** in the categorical case — sum across *classes* first,
  then average across *samples*.

## One-line summary

Cross-entropy loss measures how surprised the model was about the true answer, brutally
punishing confident wrong predictions via the log — and it's the loss every classifier and
GPT trains against.