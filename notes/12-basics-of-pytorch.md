# 12 — Basics of PyTorch

## Why this problem exists

You've built neural networks from scratch with NumPy — forward passes, backpropagation, even weight initialization. That's the right way to learn, but nobody builds real models that way. The models behind ChatGPT, Stable Diffusion, and almost every modern AI system are built with **PyTorch**, a library that does the heavy math for you and can run it on GPUs.

This problem is the handshake: it teaches you PyTorch's four core tensor operations. They're tiny individually, but you'll chain them together in every model for the rest of this course — including the final GPT.

## Where this gets used

- **Tensors** are the currency of PyTorch. Every weight, input, output, and gradient is a tensor.
- **Reshaping** shows up constantly — flattening images, folding token sequences into matrix shapes for attention, shaping batches.
- **Averaging down an axis** is how you reduce things: pooling, and the mean/variance computations behind every normalization layer (problems 13-15).
- **Concatenation** is how models join representations side by side.
- The **loss** is the thing every training loop minimizes. `loss.backward()` then `optimizer.step()` is the whole learning loop.

## The setup

NeetCode gives us four small tasks on tensors:

1. **Reshape**: turn an `M × N` tensor into `(M·N // 2) × 2` — flatten all the elements, then fold them into two columns.
2. **Column average**: average each column of a 2-D tensor (reduce along dimension 0).
3. **Concatenate**: put an `M × N` tensor next to an `M × M` tensor, producing `M × (M + N)`.
4. **MSE loss**: mean squared error between a prediction vector and a target vector.

The inputs are named `to_reshape`, `to_avg`, `cat_one`, `cat_two`, `prediction`, and `target`.

Each task is independent — solve them as four separate functions. Later problems in this section will combine these same operations inside a single model.

## Main theory

### What is a tensor?

A **tensor** is just a multi-dimensional array — essentially a NumPy array on steroids. It can live on a GPU for hardware-accelerated math, and it can remember the operations that created it so it can compute gradients later (more on that below).

On a CPU your PyTorch code behaves pretty much like NumPy. On a GPU, the *same code* runs massively in parallel across thousands of cores — which is the only reason models with billions of parameters can train at all.

The number of dimensions is the tensor's **rank**:

- rank 0 — a single number (`3.14`)
- rank 1 — a vector (`[1, 2, 3]`)
- rank 2 — a matrix (rows and columns)
- rank 3+ — a stack of matrices, which is how you store a whole *batch* of inputs

### Which axis is which

For a 2-D tensor, **dimension 0** runs down the rows and **dimension 1** runs across the columns. This matters because `mean(dim=0)` and `mean(dim=1)` give completely different answers:

- `mean(dim=0)` collapses the rows → you get one number per column (the "column average" this problem asks for).
- `mean(dim=1)` collapses the columns → one number per row.

Concretely, for `[[1, 2], [3, 4]]`:

- `mean(dim=0)` gives `[2.0, 3.0]` — i.e. `(1+3)/2` and `(2+4)/2`.
- `mean(dim=1)` gives `[1.5, 3.5]` — i.e. `(1+2)/2` and `(3+4)/2`.

Same tensor, two completely different answers. Getting this backwards is the #1 mistake in this whole section.

### Reshape: same elements, new layout

`reshape` re-arranges the same elements into a different shape. The total element count never changes — a `3 × 4` tensor has 12 elements, so it can become `6 × 2` (still 12 elements). The elements are read out row by row (row-major order), so the first 2 values become the first row, the next 2 the second row, and so on.

Concretely, the problem's example flattens its `3 × 4` block of ones into 12 ones in a row, then folds them into 6 rows of 2 — all ones, just re-arranged. If the values were `1..12`, the output would start `[[1, 2], [3, 4], ...]`.

### Concatenate: gluing tensors together

`torch.cat` joins tensors along an axis. Along `dim=1` it glues them side by side (columns get wider); along `dim=0` it stacks them on top of each other (rows get taller). Every dimension *except* the joined one must match — that's exactly why an `M × N` and an `M × M` can join into `M × (M + N)`. In the example, a `2 × 3` of ones next to a `2 × 2` of ones becomes a `2 × 5` of ones: two rows, five columns each.

Later in the course you'll glue along `dim=0` to build batches, and along `dim=1` to combine different representations of the same items — for example, adding an input embedding to a positional encoding.

### Mean squared error

MSE is the average of the squared gaps between prediction and target:

```
MSE = (1/n) Σ (y_i - ŷ_i)²
```

where `y_i` is the target and `ŷ_i` is the prediction. Squaring does two things: it removes the sign (so over- and under-predictions both count as error), and it *amplifies big misses* — an error of 2 contributes 4, an error of 4 contributes 16. That pushes the optimizer to fix its worst predictions first. MSE is the standard loss for regression; classification problems use cross-entropy instead.

The `1/n` also matters: it makes the loss an *average* rather than a sum, so the number doesn't just keep growing as you add more data points. That keeps loss values comparable no matter the batch size.

### A preview: autograd

PyTorch tensors can track their own history. If a tensor is created with `requires_grad=True`, PyTorch remembers every operation that produced it. When you call `.backward()` on a loss, it walks that history backwards and fills in `.grad` on every weight tensor — that's automatic backpropagation. This problem only uses forward-pass operations, but the gradient tracking is why the training loop later is just `loss.backward()` plus `optimizer.step()`. It's the same intuition as backpropagation (problems 08-09), just automated: instead of hand-deriving every gradient, PyTorch replays the recorded graph and fills in the numbers for you.

## The implementation

```
def reshape(to_reshape):
    return to_reshape.reshape(to_reshape.numel() // 2, 2)

def column_average(to_avg):
    return to_avg.mean(dim=0)

def concatenate(cat_one, cat_two):
    return torch.cat([cat_one, cat_two], dim=1)

def get_loss(prediction, target):
    return ((prediction - target) ** 2).mean()
```

Walkthrough:

1. `reshape` — `numel()` is the total element count (`M·N`). `// 2` gives the new row count, paired with 2 columns. A `3 × 4` input has 12 elements → `6 × 2` output, all ones as in the example.
2. `column_average` — `mean(dim=0)` averages each column down the rows. For the example, the two rows `[0.8088, 1.2614, -1.4371]` and `[-0.0056, -0.2050, -0.7201]` average to `[0.4016, 0.5282, -1.0786]` — e.g. `(0.8088 + (-0.0056)) / 2 = 0.4016`.
3. `concatenate` — `torch.cat([cat_one, cat_two], dim=1)` puts the two tensors side by side. A `2 × 3` next to a `2 × 2` becomes `2 × 5`.
4. `get_loss` — `(prediction - target)` is the per-element gap, `** 2` squares it, `.mean()` averages. For `prediction = [0, 1, 0, 1, 1]` and `target = [1, 1, 0, 0, 0]`, the gaps are `[-1, 0, 0, 1, 1]`, squared to `[1, 0, 0, 1, 1]`, averaged to `3/5 = 0.6`.

One note: `get_loss` returns a scalar tensor. If you need a plain Python float, add `.item()`.

## Where you'll see this again

- **Layer, Batch, and RMS Normalization** (problems 13-15) are mostly mean/variance computations over an axis — the same reducing you practiced here.
- **The data loader and GPT sections** reshape token sequences and batch everything into tensors.
- **The training loop** runs `.backward()` and `optimizer.step()` on tensors — the autograd preview above becomes the whole story.

## Gotchas / common mistakes

- **Wrong axis** — `mean(dim=0)` vs `mean(dim=1)` swaps rows and columns. Read the problem: "reduce along dimension 0."
- **Calling `.mean()` with no axis** — that averages every element into one number, not per-column.
- **Reshape order** — elements are read row by row, so the fold into columns is predictable. Also, `M·N` must be even for a 2-column reshape.
- **`torch.cat` shape mismatch** — all dimensions except the joined one must match, and you must pass a list (`[cat_one, cat_two]`), not two separate arguments.
- **Forgetting to square in MSE** — without `** 2` you'd get the mean absolute error, and positive and negative gaps could cancel each other out.
- **Unnecessary rounding** — the example numbers follow from plain math, so don't round unless a problem explicitly asks.
- **Using `view` instead of `reshape`** — `reshape` handles any layout; `view` can error if the tensor's memory layout isn't compatible with the new shape.

## One-line summary

Tensors are GPU-friendly arrays that remember their history for autograd, and reshape, reduce, concatenate, and MSE loss are the four building-block operations you'll chain together in every model.