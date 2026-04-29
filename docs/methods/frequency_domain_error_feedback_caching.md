# Frequency-domain error-feedback caching (implementation)

This page describes the implementation in FastCache named *frequency-domain error-feedback caching*. It is implemented on **`FastCachedTransformerBlocks`** in `xfuser/model_executor/cache/utils.py` and exposed for **Flux** through `xfuser/model_executor/cache/diffusers_adapters/flux.py` (`apply_cache_on_transformer`, `use_cache="Fast"` only).

The design is a concrete, DiT-oriented variant: **rFFT along the token sequence** (not a separate time-series axis), plus an **EMA of the spectrum of `(fresh − cached)`** fed back via **irFFT**. It can be composed with **AdaCorrection** by taking the **maximum** of the two scalar blend weights.

---

## Where it runs

| Component | Role |
|-----------|------|
| `FastCachedTransformerBlocks` | Owns flags, `_freq_error_ema` buffer, `compute_freq_event_score`, `spectral_error_feedback_residual`, `update_freq_error_ema`. |
| `process_blocks` | If `enable_adacorrection` **or** `enable_freq_error_feedback`, delegates the main double-stream path to `process_transformer_blocks`. |
| `process_transformer_blocks` | Per transformer block: always computes **cached** (`block_projections[i](h)`) and **fresh** (real block forward), then blends when the blend path is active. |
| `enhanced_process_blocks` | Same blend + EMA update when **enhanced linear approx** and/or **AdaCorrection** and/or **freq** path is used; blend branch triggers if `(enable_adacorrection or enable_freq_error_feedback) and prev_hidden_states is not None`. |

**Reference for `prev_hidden`:** blend logic compares `current_hidden` to `self.cache_context.prev_hidden_states` (updated in `get_modulated_inputs` / the block loop), i.e. the **cached “previous” hidden** used elsewhere in FastCache, not a separate tensor from the paper repo.

---

## Parameters (constructor / `apply_cache_on_transformer`)

| Name | Default | Meaning in code |
|------|---------|------------------|
| `enable_freq_error_feedback` | `False` | Master switch; when `True`, frequency score and EMA feedback are active wherever the blend branch runs. |
| `freq_event_gamma` | `2.0` | Scales the normalized FFT event score into a weight toward **fresh** (after `clamp(..., 0, 1)`). |
| `freq_error_ema_decay` | `0.85` | EMA decay \(\rho\) for the complex spectrum of `(fresh − cached)`: `ema = ρ·ema + (1−ρ)·spec`. |

AdaCorrection knobs (`enable_adacorrection`, `adacorr_gamma`, `adacorr_lambda`) are unchanged; if both are on, the scalar **`w`** is **`max(w_ada, w_freq)`** (each side already clamped to `[0,1]`).

---

## Event score (implementation)

For hidden states `current_hidden`, `prev_hidden` \(\in \mathbb{R}^{B \times P \times D}\) (batch, **token** index, channel):

1. `F_cur = torch.fft.rfft(current_hidden.float(), dim=1, norm="ortho")`  
2. `F_prev = torch.fft.rfft(prev_hidden.float(), dim=1, norm="ortho")`  
3. `diff = mean(|F_cur − F_prev|)` (scalar mean over all elements of the complex tensor)  
4. `denom = mean(|F_prev|)` with floor `1e-6`  
5. **`freq_score = diff / denom`** (float32 scalar)

So “event-driven” in code means: **larger normalized spectral change between current and cached-previous hidden states pushes the blend toward the fresh block output.**

---

## Cached vs fresh blend and feedback

For each block index `i` when the blend path is on:

- **`cached_hidden = block_projections[i](current_hidden)`**  
- **`fresh_hidden, ... = block(current_hidden, current_encoder, ...)`**  

Scalar weight:

- `w = max( clamp(adacorr_gamma * offset_score, 0, 1) [if Ada on], clamp(freq_event_gamma * freq_score, 0, 1) [if freq on] )`  
  (If only one mode is on, the other branch does not contribute.)

**Correction term** (only when freq feedback is enabled and EMA is valid):

- `corr = irfft(_freq_error_ema, n=P, dim=1, norm="ortho")` reshaped like `current_hidden`, dtype cast back to activations.  
- If `_freq_error_ema` is `None` or shape does not match `(B, P//2+1, D)` for the current `(B,P,D)`, **`corr = 0`**.

**Output:**

- `current_hidden = (1 − w) * cached_hidden + w * fresh_hidden + corr` (with `w` broadcast to tensor rank)

**EMA update** (after the above, only if `enable_freq_error_feedback`):

- `spec = rfft( (fresh_hidden − cached_hidden).detach().float(), dim=1, norm="ortho" )`  
- If `_freq_error_ema` is `None` or wrong shape: register zeros `zeros_like(spec)`  
- `_freq_error_ema.mul_(freq_error_ema_decay).add_(spec, alpha=(1 − freq_error_ema_decay))`

Then `prev_hidden` used inside the per-layer loop is set from the **new** `current_hidden` for the next block (see `process_transformer_blocks`).

---

## Scope and limitations (as implemented)

- **Adapter:** Only the Flux `create_cached_transformer_blocks` / `apply_cache_on_transformer` path passes these kwargs into `FastCachedTransformerBlocks`. Other cache types (`Fb`, `Tea`) do not receive this API in `flux.py`.  
- **Axis:** FFT is always on **token dimension `dim=1`**. For DiTs, that is the flattened patch / joint-attention token order, not necessarily a physical “time” axis.  
- **State:** One `_freq_error_ema` per wrapper module; shape must match current `(B, P, D)` or correction is skipped until the buffer is reinitialized.  
- **Cost:** The blend path **always runs the real transformer block** to obtain `fresh_hidden` whenever AdaCorrection or freq feedback is enabled (same pattern as AdaCorrection-only).

---

## Example usage

**Script** (reloads the pipeline between runs so patches are not stacked):

```bash
python examples/test_freq_error_feedback.py \
  --model "black-forest-labs/FLUX.1-schnell" \
  --num_inference_steps 30 \
  --freq_event_gamma 2.0 \
  --freq_error_ema_decay 0.85 \
  --enable_adacorrection
```

**API** (minimal):

```python
from xfuser.model_executor.cache.diffusers_adapters import apply_cache_on_transformer

apply_cache_on_transformer(
    transformer,
    use_cache="Fast",
    rel_l1_thresh=0.05,
    motion_threshold=0.1,
    return_hidden_states_first=False,
    num_steps=30,
    enable_freq_error_feedback=True,
    freq_event_gamma=2.0,
    freq_error_ema_decay=0.85,
)
```

---

## Citation

If you use this caching method, cite:

```bibtex
@misc{liu2026frequencydiffusioncaching,
      title={Accelerating Frequency Domain Diffusion Models with Error-Feedback Event-Driven Caching}, 
      author={Dong Liu and Haisheng Wang and Yanxuan Yu},
      year={2026},
      eprint={2604.22901},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.22901}, 
}
```

FastCache-xDiT and related papers are listed under [Cite Us](../../README.md#cite-us) in the root `README.md`.
