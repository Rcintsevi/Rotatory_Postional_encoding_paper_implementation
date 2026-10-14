# RoPE in a small GPT (RoFormer reproduction, reduced scale)

Re-implementation of **Rotary Position Embedding** from
*RoFormer: Enhanced Transformer with Rotary Position Embedding* (Su, Lu, Pan, Murtadha, Wen, Liu, arXiv:2104.09864)
inside a small character-level GPT, compared against sinusoidal and learned absolute positions.

## What is in the repo

| File | Purpose |
|---|---|
| `RoPE_GPT.ipynb` | The whole project: data, models, training, evaluation, ablations, plots (outputs included) |
| `results.json` | All numbers produced by the notebook (loss curves, perplexities per length) |
| `figs/` | Figures used in the report |
| `report.pdf` | 4-6 page IEEE-style report |

## Setup

```
pip install torch matplotlib
```
Python 3.9+; no other dependencies. The notebook downloads TinyShakespeare itself
(`karpathy/char-rnn`, ~1.1 MB). To run in Colab: open the notebook, set Runtime -> GPU, Run all.

## How to run

Open `RoPE_GPT.ipynb` and *Run all*. The notebook trains six models one after another:
`sinusoidal`, `learned`, `rope` (main comparison) and `rope_base100`, `rope_base1e6`, `nope` (ablations).
Training uses fixed seed `1337`, and a dedicated generator so every model sees the *same* training batches.

Compute used for the reported numbers: 2 CPU cores, ~16 min per model, 3000 steps each.
On a Colab GPU the same notebook finishes in a few minutes; you can raise `max_steps` in the `cfg` cell
for lower perplexity.

## Core code (what you must be able to explain)

`build_rope_cache`, `rotate_pairs`, `apply_rope` (Section 3 of the notebook) implement paper Eq. 15 / 34.
They are applied to queries and keys only, inside `CausalSelfAttention.forward`.
The cell right after them numerically checks that `<R_m q, R_n k>` depends only on `n - m`.

## Results

See the results table printed in Section 8 of the notebook and Section IV of `report.pdf`.

## Sources / references consulted

* The RoFormer paper (arXiv:2104.09864). All code was written from the equations in the paper.
* **No official or third-party RoPE repository was consulted for the implementation.**
* TinyShakespeare data file from `karpathy/char-rnn` (data only, no code used).
* Standard PyTorch building blocks only: `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.GELU`, `AdamW`.
  Attention (masking, softmax, scaling) is written by hand.
