# 06 — Linear Regression (Training)

## Why this problem exists

Last problem you built the *forward pass* — you could make predictions and measure their
error. But a model that can't improve is useless. This problem is where the model actually
**learns**: it closes the loop and ties together everything from this entire series.

The training loop is the same three steps every neural network repeats, from a single
neuron all the way up to GPT:

1. Forward pass — predict using current weights.
2. Compute gradients — how much each weight contributed to the error.
3. Update weights — nudge each weight to reduce the error.

That's it. Learn this loop and you've learned how all of ML trains.

## Where this gets used

- **Everything.** The `W ← W - α · ∇L` update is the universal learning step.
- In PyTorch later, `optimizer.step()` does exactly this.
- For **GPT**: the same three-step loop, repeated over billions of weights, is how the
  whole model learns.

## The setup

Implement `train_model(X, Y, num_iterations, initial_weights)`, which:

- `X` — feature matrix, `n` rows, each with 3 features (`X[i].length = 3`).
- `Y` — the `n` target values.
- `num_iterations` — how many gradient-descent steps to run (always `> 0`).
- `initial_weights` — the starting `[w_1, w_2, w_3]`.

Returns a NumPy array of 3 elements — the optimized weights after all iterations.

NeetCode hands you a helper `get_derivative()` that returns the gradient of the loss with
respect to each weight. (In practice, ML engineers almost never derive gradients by hand —
later you'll rely on PyTorch's autograd to do it automatically.)

## Main theory

### The update rule (gradient descent, again)

Each iteration nudges the weights by subtracting the gradient scaled by a learning rate:

```
W ← W - α * ∇_W L
```

- `∇_W L` is the gradient: a vector of partial derivatives saying how much the loss
  changes as each weight wiggles. Provided by `get_derivative()`.
- `α` (alpha) is the **learning rate**, already set to `0.01` as `self.learning_rate`. It
  controls step size — too big and you overshoot the minimum, too small and training takes
  forever.
- The **minus sign** moves *against* the slope, exactly like the `x = x - lr * slope` loop
  from problem 01. This is the same idea, just with 3 weights and a loss instead of `x²`.

### Why each weight gets its own gradient

In `y_hat = w_1*x_1 + w_2*x_2 + w_3*x_3`, each weight influences the prediction differently,
depending on how big its feature is. The gradient tells you, per weight, which direction
(and roughly how far) to move to shrink the loss. So the update is applied per-weight:
each `w` becomes `w - α * (∂L/∂w)`.

### Why we repeat

One nudge rarely lands you at the bottom. Each step, `get_derivative()` is recomputed with
the *current* weights, because as the weights change, the correct direction changes too.
Running `num_iterations` steps walks you down toward the minimum — smaller loss each time.

## The implementation

```python
import numpy as np

class LinearRegression:
    def __init__(self):
        self.learning_rate = 0.01

    def get_derivative(self, model_prediction, ground_truth, X):
        # provided by NeetCode — returns gradient of MSE wrt each weight
        ...

    def train_model(self, X, Y, num_iterations, initial_weights):
        weights = np.array(initial_weights, dtype=float)
        X = np.array(X)
        Y = np.array(Y)
        for _ in range(num_iterations):
            pred = np.dot(X, weights)          # 1. forward pass
            derivative = self.get_derivative(pred, Y, X)   # 2. gradient
            weights = weights - self.learning_rate * derivative   # 3. update
        return weights
```

Walkthrough:

1. `np.array(initial_weights, dtype=float)` — keep weights as a NumPy array of 3 floats so
   the vectorized update works.
2. `pred = np.dot(X, weights)` — the forward pass from problem 05: predictions from current
   weights.
3. `self.get_derivative(pred, Y, X)` — the gradient of the loss with respect to each
   weight, given current predictions. Recomputing it every iteration is what makes the
   steps correct.
4. `weights = weights - self.learning_rate * derivative` — the single update rule,
   applied to all 3 weights at once (array math). This is the minus-scaled-gradient step.
5. `num_iterations` times — enough steps to converge.
6. Return the final `weights` — the learned model.

**Sanity-check the example.** `X = [[1,2,3],[1,1,1]]`, `Y = [6,3]`, 10 iterations,
`initial_weights = [0.2, 0.1, 0.6]` → the loop returns `[0.50678, 0.59057, 1.27435]`. The
weights moved from their starting values toward ones that predict `Y` well.

## Where you'll see this again

- **Training loop** (problem 12) generalizes this exact three-step structure to real
  networks.
- **Backpropagation** (problems 08-09) is the smart way to compute all those gradients for
  a deep network at once — the thing `get_derivative()` does by hand here, automated.
- **PyTorch**: `loss.backward()` computes the gradient, `optimizer.step()` applies the
  update. Same loop, automated.

## Gotchas / common mistakes

- **Forgetting to recompute the gradient each iteration** — reuse stale gradients and you
  stop following the current slope.
- **Wrong sign** — it's `W - α·∇L`, subtracting the gradient. Adding walks uphill.
- **Forgetting to convert to a NumPy array** — mixing lists and `np.dot` silently gives
  wrong shapes or errors.
- **Applying the update inside a list/array vs element-wise** — with 3 weights, use array
  math so all weights update at once.
- **Hard-coding `get_derivative`** — it's provided; the task is the *loop*, not the math.

## One-line summary

Training = repeating forward pass → compute gradient → `W = W - α·∇L` until the weights
stop improving, and this three-step loop is how every neural network (up to GPT) learns.