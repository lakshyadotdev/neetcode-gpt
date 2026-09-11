# 01 — Gradient Descent

## Why this problem exists

A model's job is to make good predictions. It does that using **weights** — numbers it
multiplies its inputs by. The problem: how do you find the *right* weights?

You define a **loss function** (a measure of how wrong the predictions are) and then the
whole of learning becomes one question:

> Find the weights that make the loss as small as possible.

For simple problems you could solve this with algebra, but for real models (neural nets,
GPT) there is no formula for the best weights — the loss is far too complicated. So we
use a general, iterative method: **gradient descent**. Start somewhere, look at the slope,
take a step downhill, repeat. That's it. Everything else in this course is this one loop.

## Where this gets used

- Every neural network ever trained (image classifiers, GPT, everything) uses gradient
  descent or a variant of it (SGD, Adam — same idea, smarter steps).
- When you later call `optimizer.step()` in PyTorch, this is literally what happens
  under the hood.
- "Training" = running gradient descent on the loss until the loss stops shrinking.

## The setup

NeetCode gives us the simplest possible function to practice on:

```
f(x) = x²
```

We want to find the `x` that minimizes it. You already know the answer is `x = 0`, but
we're going to *find* it mechanically, the way a machine would.

## Main theory

### The derivative is the slope

The **derivative** `f'(x)` tells you the slope of `f` at point `x` — i.e. if you move a
tiny bit to the right, does the function go up or down, and how steeply?

- `f'(x) > 0` → function is rising as you go right.
- `f'(x) < 0` → function is falling as you go right.
- `f'(x) = 0` → flat spot (could be a minimum, maximum, or plateau).

For `f(x) = x²`, the derivative is:

```
f'(x) = 2x
```

At `x = 3`, slope is `6` (steep uphill). At `x = -2`, slope is `-4` (going down as you
move right). At `x = 0`, slope is `0` (the bottom of the bowl).

### The update rule

To decrease `f`, you want to move **against** the slope:

- Slope positive → the function is higher to your right → move **left** (subtract).
- Slope negative → the function is lower to your right → move **right** (add).

Both cases are covered by one formula:

```
x_new = x - learning_rate * f'(x)
```

The `learning_rate` (a small number like `0.1`) just scales how big a step you take —
you don't want to jump all the way to the top of the opposite slope.

### Why this actually works (the tangent-line argument)

At any point, `f'(x)` is the slope of the **tangent line** to the curve. The tangent is a
local straight-line approximation of the curve. Walking *opposite* to the tangent's slope
is, locally, walking downhill along that line. Take a small enough step and you'll end up
at a lower point on the real curve. Repeat — each step lands you lower until you reach
the bottom.

### Convergence, concretely

For `f(x) = x²` the update becomes:

```
x_new = x - lr * 2x = x(1 - 2·lr)
```

So **every step multiplies the current position by the same factor** `(1 - 2·lr)`:

- `lr = 0.1` → each step multiplies by `0.8` → 3, 2.4, 1.92, ... shrinking toward 0.
- `lr = 0.5` → factor `0` → you land exactly on 0 in one step.
- `lr = 0.6` → factor `-0.2` → you overshoot and land at -0.6, +0.12, ... it still
  converges because the factor's size is < 1.
- `lr > 1` → factor's size `> 1` → each step bounces further away → **diverges, blows up**.

That's the whole story of "learning rate too big": the step is so large you overshoot
past the bottom, and if it's large enough you overshoot *more* each time.

### Multiple parameters (preview of real ML)

Real models have thousands to billions of weights. The same rule applies per weight: for
each weight, compute how much the loss changes when *that* weight wiggles (this is the
**partial derivative** / gradient for that weight), then nudge it opposite that slope.
The full set of slopes is called **the gradient**, a vector that points in the steepest
uphill direction — hence *gradient* descent. The minus sign in the update moves you in
the steepest *downhill* direction.

### Local minima (the catch)

Gradient descent finds the bottom of the *valley you're in*. If the loss surface has
multiple valleys (true for real neural nets), you might get stuck in a local valley that
isn't the global best. Real training accepts this and works around it (random starts,
momentum, adaptive learning rates).

## The implementation

```
def get_minimizer(iterations, learning_rate, init) -> float:
    x = init
    for _ in range(iterations):
        slope = 2 * x          # f'(x)
        x = x - learning_rate * slope
    return round(x, 5)
```

Walkthrough of what it does:

1. `slope = 2 * x` — derivative at the current spot.
2. `x = x - learning_rate * slope` — the one update rule, repeated.
3. `iterations` times — enough steps to get close to the bottom.
4. `round(..., 5)` — NeetCode expects an answer rounded to 5 decimals.

Note: you could keep the slope fixed at `2 * init` and it'd still roughly work, but
recomputing it each step is what makes gradient descent correct — as `x` changes, the
slope changes too, and you must follow the *current* slope.

## Where you'll see this again

- **Everything in Math Foundations** builds on this loop.
- **Linear regression training** (problem 06): same loop, but the "slope" is the
  derivative of a loss with respect to two weights, and the function is a line, not `x²`.
- **Backpropagation** (problems 08-09): the smart way to compute all those slopes for a
  whole neural net at once.
- **Training loop / GPT**: the exact same `x = x - lr * gradient` applied to billions
  of weights with fancy optimizers.

## Gotchas / common mistakes

- **Forgetting the minus sign** — adding instead of subtracting walks *uphill*.
- **Learning rate too large** — diverges instead of converging (see the `1 - 2·lr` factor).
- **Not updating the slope** — recompute `f'(x)` after every step, using the *new* `x`.
- **Rounding too early** — round only the final result, not inside the loop.

## One-line summary

Gradient descent = repeatedly take small steps opposite the slope of the loss, until you
reach the bottom — and that loop is how all of ML learns.