# RoPE (Rotary Position Embedding) in a Small GPT

A from-scratch reproduction of **RoFormer: Enhanced Transformer with Rotary Position Embedding**
(Su, Lu, Pan, Murtadha, Wen, Liu — [arXiv 2104.09864](https://arxiv.org/abs/2104.09864)),
implemented in plain PyTorch on a character-level GPT trained on TinyShakespeare.

The notebook compares three position-encoding methods inside the *same* model, with the same
optimiser, seed and training batches, so the only variable is how position information is injected.

| Method | Where it acts | Paper reference |
|---|---|---|
| `sinusoidal` | Fixed absolute encoding added to token embeddings | Eq. 4 |
| `learned` | Trainable absolute embedding table added to token embeddings | Eq. 3 |
| `rope` | Rotation applied to queries and keys inside attention | Eq. 14-16, 34 |

## What the notebook does

1. Implements sinusoidal, learned and rotary position encodings with hand-written causal attention
   (no flash-attention or `scaled_dot_product_attention`).
2. Verifies RoPE's key properties numerically: rotation preserves vector norm, and the attention
   score depends only on the **relative** distance `n - m`.
3. Trains each variant on TinyShakespeare and compares validation loss / perplexity
   (a reduced-scale analogue of the paper's Fig. 3, left).
4. Tests **length extrapolation** by evaluating on contexts up to 4x the training length.
5. Runs **ablations**: RoPE base frequency (100 and 1,000,000 vs. the default 10,000) and
   **NoPE** (no position information at all).
6. Reproduces RoPE's **long-term decay** property (paper Fig. 2).

## Setup

- Python 3 with `torch` and `matplotlib`
- Runs on CPU or GPU (the reported runs used CUDA; each model trains in roughly 1-2 minutes)
- TinyShakespeare is downloaded automatically on first run
- On Colab: *Runtime -> Change runtime type -> GPU*, then *Run all*

```bash
pip install torch matplotlib
jupyter notebook RoPE_GPT.ipynb
```

Outputs are written to `figs/` (plots), `ckpt/`, and `results.json` (all metrics).

## Model and training configuration

| Setting | Value |
|---|---|
| Layers / heads / d_model | 4 / 4 / 128 (head_dim = 32) |
| Training context (`block_size`) | 128 |
| Batch size | 32 |
| Steps | 3000 |
| Optimiser | AdamW, lr 2e-3 -> 2e-4 (100-step warmup, cosine decay), weight decay 0.01 |
| Dropout / grad clip | 0.1 / 1.0 |
| Data | TinyShakespeare, 90/10 train/val split, vocab size 65 |
| Seed | 1337 |
| Parameters | 810,049 (sinusoidal, RoPE, NoPE) / 826,433 (learned) |

## How RoPE is implemented

Each query/key vector is split into `d/2` pairs `(x_{2i}, x_{2i+1})`. Pair `i` at position `m` is
rotated by the angle `m * theta_i`, where `theta_i = base^(-2i/d)`:

```python
def apply_rope(x, cos, sin):          # x: (B, n_head, T, head_dim)
    return x * cos + rotate_pairs(x) * sin
```

- Only **q and k** are rotated; **v** is untouched, and nothing is added to the embeddings.
- `cos`/`sin` tables are precomputed once up to `MAX_LEN = 1024` and stored as non-persistent buffers.
- Because `R_m^T R_n = R_{n-m}`, the score `q_m . k_n` depends only on the relative offset `n - m`.

## Results

Validation **perplexity** (character level), training context = 128 (`-` = undefined, learned
embeddings have no vectors beyond `block_size`):

| Model | 32 | 64 | **128** | 192 | 256 | 384 | 512 |
|---|---|---|---|---|---|---|---|
| sinusoidal | 5.209 | 5.010 | 4.903 | 8.085 | 10.957 | 14.521 | 16.869 |
| learned | 5.214 | 5.026 | 4.924 | - | - | - | - |
| **rope** | **4.927** | **4.708** | **4.594** | **4.889** | **6.721** | 11.341 | 15.629 |
| rope (base 100) | 4.950 | 4.734 | 4.624 | 5.318 | 7.663 | 12.403 | 16.457 |
| rope (base 1e6) | 4.946 | 4.731 | 4.619 | 6.789 | 9.932 | 15.706 | 20.609 |
| nope | 6.090 | 5.885 | 5.825 | 6.142 | 6.920 | 8.413 | 9.469 |

Final validation loss at L = 128: sinusoidal 1.5899, learned 1.5942, **RoPE 1.5248**,
RoPE base 100 1.5312, RoPE base 1e6 1.5302, NoPE 1.7622.

Long-term decay of RoPE (mean of the relative upper bound, d = 256): 45.37 at distance 1-10,
18.94 at 100-110, 13.99 at 240-255 — it decays as the paper predicts.

### Key takeaways

- **RoPE gives the best in-distribution perplexity** (4.594 vs. 4.903 sinusoidal and 4.924 learned)
  with no extra parameters, and converges faster early in training.
- **Extrapolation:** RoPE stays close to its training-length quality at 1.5x (4.889) and degrades
  much more slowly than sinusoidal at 2x (6.721 vs. 10.957). Beyond ~3x all absolute and rotary
  variants degrade substantially.
- **Base frequency matters:** the default 10,000 is best; base 1e6 extrapolates worst here.
- **NoPE** is clearly worse in-distribution (5.825) but degrades most gracefully at long lengths,
  a reminder that a causal mask leaks some positional information and that RoPE is not a
  free pass to unbounded context.

> Note: the per-step validation numbers printed during training use 100 validation chunks, while
> the final numbers use the full validation set, so they differ slightly.

## Limitations of this reproduction

- Tiny model, character-level data, one seed — differences of a few hundredths in loss are
  within plausible run-to-run noise; the large gaps (RoPE vs. NoPE, extrapolation) are the
  reliable signals.
- The paper's experiments (machine translation, GLUE fine-tuning, long-document tasks) are not
  reproduced.

## Repository layout

```
RoPE_GPT.ipynb   # full experiment: model, training, ablations, plots
results.json     # metrics for every run (generated)
figs/            # loss_curves.png, extrapolation.png, position_loss.png, decay.png
```

## Reference

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu.
*RoFormer: Enhanced Transformer with Rotary Position Embedding.* arXiv:2104.09864.