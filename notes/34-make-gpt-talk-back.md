# 34 — Make GPT Talk Back

## Why this problem exists

Your GPT can train. Now make it speak. You started with gradient descent on a quadratic,
built neurons, learned backprop, stacked networks, processed language, implemented
attention, assembled a transformer, trained it — and now it finally generates text. The
loop you'll write here is *exactly* what runs every time someone talks to ChatGPT:
predict the next token, append it, repeat.

This is the final "Generate" step of the pipeline:

```
Tokenize → Data → Model → Train → Generate
```

## Where this gets used

- Every single response from ChatGPT, Claude, or any LLM is this autoregressive loop,
  repeated thousands of times.
- The model you generate with here was trained on Drake lyrics — you get to see a tiny
  real model produce actual sentences.
- This loop is also the reason the *next* problem (KV-Cache) exists: generating one token
  at a time is slow, and this is what gets optimized.

## The setup

You're given a pre-trained GPT, `new_chars` (how many tokens to produce), `context` (the
seed sequence — usually just a newline token to kick things off), `context_length` (the
attention window), and `int_to_char` (maps token IDs back to characters). Return the full
generated text as a string.

```
new_chars = 60, context = [[0]], context_length = 128
int_to_char = {0: '\n', 1: ' ', 2: '!', 3: 'a', 4: 'b', ...}
Output: "Yeah I'm just never playing without you Chorus: Drake and Futur"
```

The model only sees the last `context_length` tokens, so you must crop older ones at
every step.

## Main theory

### The autoregressive generation loop

Text generation is **autoregressive**: the model produces one token at a time, and each
new token's prediction depends on everything generated so far. Like writing a sentence
word by word, where each word choice depends on all the previous words. The loop:

```
1. logits = model(context)          # (1, seq_len, vocab_size)
2. logits = logits[:, -1, :]        # keep only the LAST position
3. probs  = softmax(logits)         # probability over every token
4. next   = sample from probs       # torch.multinomial
5. append next to context, crop to context_length, repeat
```

Step 2 is the key trick: the model already produced scores for *every* position in the
context, but the only scores we care about are at the final position — those say "given
everything so far, what comes next?"

### Greedy vs sampling

Once you have probabilities, how do you pick the token? Two families of strategies:

- **Greedy**: always pick the token with the highest probability (`argmax`). Deterministic
  and easy, but repetitive and dull — the model gets stuck in loops and never surprises.
- **Sampling**: treat the probabilities as a lottery and draw one token randomly.
  Lower-probability tokens still get picked occasionally, which is what makes generated
  text feel alive.

This problem uses sampling (`torch.multinomial`), which picks a token with probability
proportional to the distribution you give it.

### Temperature

Temperature reshapes the distribution *before* softmax by dividing each logit:

```
p_i = exp(z_i / T) / Σ_j exp(z_j / T)
```

- `T < 1` (e.g. 0.5): divides logits, then exponentiates → the big scores get relatively
  even bigger → distribution is **sharper**, more confident, closer to greedy.
- `T > 1` (e.g. 1.5): flattens the differences → distribution is **wider**, more random,
  more creative.
- `T → 0`: collapses to greedy (the max wins almost surely).
- `T = 1`: no change — plain softmax.

That's the knob ChatGPT exposes as "creativity". The model here uses the plain `T = 1`
version (softmax, then multinomial).

### Top-k and top-p

These are guardrails on top of sampling:

- **Top-k**: zero out everything except the `k` most likely tokens, then sample. Stops
  the model from ever picking a ridiculous word.
- **Top-p** (nucleus sampling): sort tokens by probability and keep the *smallest set
  whose cumulative probability reaches `p`* (e.g. 0.9), then sample from those.

Both prevent the model from sampling extremely unlikely tokens while keeping some
randomness.

### The context window is a hard limit

Attention can only look back `context_length` tokens. If generation passes that length,
the oldest tokens must be dropped — otherwise the model either errors or attends to more
than it was built for. Cropping each step is what keeps generation correct over long
outputs.

## The implementation

```
import torch
import torch.nn.functional as F

def generate(model, new_chars, context, context_length, int_to_char):
    model.eval()
    generated = []
    for _ in range(new_chars):
        # crop to the last context_length tokens
        x = context[:, -context_length:]

        with torch.no_grad():
            logits = model(x)                # (1, seq_len, vocab_size)
        logits = logits[:, -1, :]            # (1, vocab_size)
        probs = F.softmax(logits, dim=-1)

        next_idx = torch.multinomial(probs, num_samples=1).item()
        generated.append(int_to_char[next_idx])

        # append the new token to the context
        context = torch.cat(
            [context, torch.tensor([[next_idx]], device=context.device)],
            dim=1,
        )

    return "".join(generated)
```

Walkthrough:

1. `model.eval()` — switches off dropout/training behavior. We're not learning here.
2. `context[:, -context_length:]` — drop everything older than the window. Python
   negative slicing keeps the *last* `context_length` tokens.
3. `torch.no_grad()` — generation doesn't need gradients, so we skip building the
   computation graph. Faster and uses less memory.
4. `logits[:, -1, :]` — take only the final position's scores (shape `(1, vocab)`).
5. `F.softmax(logits, dim=-1)` — turn scores into a probability distribution over all
   tokens (the dim=-1 matters: softmax across the vocabulary, not across the batch).
6. `torch.multinomial(probs, num_samples=1)` — draw one token from that distribution.
   `.item()` turns the 1-element tensor into a plain int so it can index `int_to_char`.
7. Append the token ID back onto the context so the next iteration sees it — build the
   new tensor on the *same device* as the context.
8. Repeat exactly `new_chars` times (one token per iteration — no off-by-one) and join
   the characters into one string.

## Where you'll see this again

- **KV-Cache** (next problem): this loop is painfully slow — every step recomputes
  attention over the whole history, and the cache exists to fix exactly that.
- **GQA** (problem 36): shrinks the memory the cache needs.
- In real tools (nanoGPT, llama.cpp), generation is this loop plus temperature, top-k,
  top-p, and a KV-cache.

## Gotchas / common mistakes

- **Not cropping the context** — past `context_length` tokens, the model breaks (wrong
  shape or attends to too much).
- **Taking the wrong position** — must be the *last* position `[:, -1, :]`; using all
  positions (or the first) samples nonsense.
- **Picking greedy instead of sampling** — the spec wants `torch.multinomial`; argmax
  gives repetitive text and fails the tests.
- **Device mismatch** when appending the new token — build it with `device=context.device`.
- **Indexing `int_to_char` with a tensor** — convert with `.item()` first.
- **Softmaxing the wrong dim** — must be `dim=-1` (across the vocab).
- **Forgetting `no_grad` / `model.eval()`** — not wrong, but slow (and dropout changes
  behavior).

## One-line summary

To make a GPT talk: feed it the seed, read the last position's logits, softmax them into
probabilities, sample one token, append it to the context, crop to the window, and repeat
— that autoregressive loop is how every LLM writes text.