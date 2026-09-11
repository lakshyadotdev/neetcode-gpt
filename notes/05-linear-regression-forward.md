# 05 — Linear Regression (Forward)

## Why this problem exists

You now have the whole toolkit: gradient descent to optimize, activations to add
non-linearity, softmax for probabilities, cross-entropy for error. This problem is where
you finally *put it together* — and it's a deceptively small step.

Linear regression is the "Hello World" of machine learning: predict a number from features
using a straight-line relationship. And here's the key insight: **a linear regression model
is literally a single neuron with no activation function** (no sigmoid, no ReLU — the sum
comes straight out). A neural network is just many of these stacked together.

This problem gives you the **forward pass**: given features and weights, make a prediction,
then measure how wrong you were.

## Where this gets used

- It's the base of every regression problem (predicting prices, temperatures, any
  continuous number).
- Every neural network's forward pass is the same idea — multiply features by weights and
  sum — just repeated across many layers and many neurons.
- The error metric here, **Mean Squared Error**, is one of the two big losses (alongside
  cross-entropy). You'll see it again throughout the course.

## The setup

Implement two functions:

1. `get_model_prediction(X, weights)` — the forward pass:
   ```
   Y_hat = X · W      (dot product of feature matrix and weight vector)
   ```
2. `get_error(model_prediction, ground_truth)` — Mean Squared Error between predictions
   and the real answers:
   ```
   MSE = (1/n) * sum over i of (y_hat_i - y_i)^2
   ```

`X` is a matrix: each of the `n` rows is one data point with `m` features. `weights` is a
list of `m` numbers, one per feature.

## Main theory

### The forward pass is a dot product

Linear regression assumes the answer is a weighted sum of the features:

```
y_hat = w_1 * x_1 + w_2 * x_2 + ... + w_m * x_m
```

For the Uber-price example from the spec: with features time, distance, duration, the model
predicts:

```
price = w_1 * time + w_2 * distance + w_3 * duration
```

No exponents, no logs, just weighted sums. `X · W` means: for each row, multiply every
feature by its matching weight and add them up. That's a single neuron doing its weighted
sum — minus any activation.

### Mean Squared Error

Once you have predictions, you need to know how wrong they are. MSE does this per sample:

- Take the difference `y_hat_i - y_i` (prediction minus truth).
- **Square** it — this punishes big errors extra hard, and kills any negative signs
  (a miss of +2 and -2 both cost the same).
- Average over all `n` samples.

The **squared** term is what makes it "mean squared error": because it squares, one huge
mistake costs more than several small ones, which is usually what you want a model to care
about.

**Concrete check.** `model_prediction = [1.0, 2.0, 3.0]`, `ground_truth = [1.5, 2.5, 2.0]`:

```
((1.0-1.5)^2 + (2.0-2.5)^2 + (3.0-2.0)^2) / 3
= (0.25 + 0.25 + 1.0) / 3
= 0.5
```

The last sample is the biggest miss (`1.0` squared), so it dominates the total.

## The implementation

```python
import numpy as np

def get_model_prediction(X, weights):
    return np.dot(X, weights)

def get_error(model_prediction, ground_truth):
    return np.mean((model_prediction - ground_truth) ** 2)
```

Walkthrough:

1. `np.dot(X, weights)` — the dot product. Each row of `X` is multiplied by `weights`
   and summed, giving one prediction per row. That's the whole forward pass.
2. `model_prediction - ground_truth` — the per-sample difference (element-wise, arrays of
   equal length `n`).
3. `( ... ) ** 2` — square each difference. This is the "S" in MSE.
4. `np.mean(...)` — the `1/n` average. This is the "M" and the "E" together.

Note that `get_error` is deliberately separate from the forward pass: you can measure the
error of *any* set of predictions, no matter how they were produced.

## Where you'll see this again

- **Linear regression training** (problem 06): the exact same forward pass + MSE, but now
  wrapped in a gradient-descent loop that updates the weights to shrink the error.
- **Single neuron / MLP** (problems 07-10): this forward pass becomes the first layer of a
  real network, with an activation added.
- **Training loop / GPT** (later): the same predict-then-measure-error structure, just
  with billions of weights.

## Gotchas / common mistakes

- **Forgetting the bias/intercept** — for this problem it's just `X · W` with no bias
  term; don't add one that isn't in the spec.
- **Mis-shaping** — `weights` must have length `m` matching the columns of `X`, or the dot
  product fails.
- **Squaring then averaging** — MSE must square *before* averaging, not average then square.
- **Forgetting the `** 2` entirely** — that would just be mean absolute error, not MSE.
- **Looping over rows** instead of using `np.dot` — vectorized dot is the point.

## One-line summary

Linear regression's forward pass is a single neuron without an activation — `X · W` — and
MSE measures its error by squaring each miss and averaging, ready for the training loop
next.