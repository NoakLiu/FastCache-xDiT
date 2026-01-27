# AdaCorrection: Adaptive Offset Cache Correction for Accurate Diffusion Transformers

AdaCorrection is a lightweight, training-free framework for adaptive offset cache correction in Diffusion Transformers (DiTs). Unlike static cache reuse strategies, AdaCorrection dynamically detects offset drift and adaptively corrects stale activations layer by layer and step by step, maintaining high generation fidelity while enabling efficient cache reuse.

## Core Innovation: Adaptive Offset Correction

The fundamental insight behind AdaCorrection is that cached activations can become misaligned with current diffusion dynamics, leading to semantic drift and quality degradation. Instead of using fixed reuse schedules, AdaCorrection introduces **adaptive offset correction** that:

1. **Detects Misalignment**: Measures spatio-temporal offset drift per token and per layer
2. **Adaptive Correction**: Computes correction weights based on detected misalignment
3. **Quality-Preserving Blending**: Interpolates between cached and fresh computations proportionally to offset magnitude

## Technical Framework: Two-Module Architecture

AdaCorrection consists of two key modules that work in tandem:

### 1. Offset Estimation Module (OEM)

The OEM quantifies misalignment between current and cached activations using spatio-temporal deviation statistics.

**Temporal Deviation:**
```
Δ_temp(t) = (1/BP) * Σ_{b,i} ||h_t^l[b,i,:] - h_{t-1}^l[b,i,:]||_2
```

This measures how much each token has changed across timesteps, capturing motion and temporal dynamics.

**Spatial Variation:**
```
Δ_spatial(t) = (1/BP) * Σ_{b,i} sqrt(Var_d(h_t^l[b,i,d]))
```

This measures channel-wise dispersion within each token, reflecting structural complexity and spatial variation.

**Offset Score:**
```
S_t^l = ||Δ_temp(t)||^2 + λ * ||∇_x h_t^l||^2
```

Where `||∇_x h_t^l||^2` is approximated by `Δ_spatial(t)` in practice. The offset score `S_t^l` reflects both temporal change and spatial complexity—two indicators of when stale cache reuse may induce semantic degradation.

### 2. Adaptive Correction Module (ACM)

The ACM converts misalignment signals into correction weights that govern interpolation between cached and fresh computations.

**Correction Weight:**
```
λ_t^l = clip(γ * S_t^l, 0, 1)
```

Where:
- `γ` (gamma) is the sensitivity parameter that controls adaptation rate
- Large `γ` makes the system more responsive to offset changes
- `λ_t^l → 1` when offset is large (strong correction, rely on fresh computation)
- `λ_t^l → 0` when offset is small (allow cache reuse)

**Quality-Aware Blending:**
```
ĥ_{t,l+1} = (1 - λ_t^l) * h̃_{t,l+1} + λ_t^l * h_{t,l+1}
```

Where:
- `h̃_{t,l+1}` is the cached activation (from linear approximation or previous timestep)
- `h_{t,l+1}` is the freshly computed activation
- `ĥ_{t,l+1}` is the corrected output

This approach prioritizes quality preservation by ensuring that significant offsets trigger immediate correction while minor offsets allow efficient reuse.

## Algorithm: AdaCorrection Inference

```
Algorithm: AdaCorrection Inference with Quality-Preserving Correction
Input: latent z_t, step t, cacheable layers L, sensitivity γ, spatial weight λ
Output: corrected hidden states ĥ_t^L

1. For each layer ℓ = 0 to L-1:
   a. Compute temporal deviation: Δ_temp(t) = (1/BP) * Σ ||h_t^l[b,i,:] - h_{t-1}^l[b,i,:]||_2
   b. Compute spatial variation: Δ_spatial(t) = (1/BP) * Σ sqrt(Var_d(h_t^l[b,i,d]))
   c. Compute offset score: S_t^l = ||Δ_temp(t)||^2 + λ * Δ_spatial(t)
   d. Compute correction weight: λ_t^l = clip(γ * S_t^l, 0, 1)

2. For each layer ℓ = 0 to L-1:
   a. If cache is available:
      - Get cached result: h̃_{t,l+1} = cache[ℓ].out
      - Compute fresh result: h_{t,l+1} = Block_ℓ(h_t^l, t)
      - Blend: ĥ_{t,l+1} = (1 - λ_t^l) * h̃_{t,l+1} + λ_t^l * h_{t,l+1}
   b. Else:
      - Full recomputation: ĥ_{t,l+1} = Block_ℓ(h_t^l, t)
   c. Update cache[ℓ] ← (h_t^l, ĥ_{t,l+1}, t)

3. Return Decode(ĥ_t^L)
```

## Key Advantages

- **Quality Preservation**: Maintains near-original FID scores (4.37 vs 4.42 baseline, only 0.05 difference)
- **Training-Free**: No model retraining or architecture modification required
- **Plug-and-Play**: Composes cleanly with existing caching methods (FastCache, TeaCache, AdaCache, etc.)
- **Adaptive**: Dynamically adjusts correction based on detected misalignment
- **Efficient**: Minimal computational overhead while improving cache hit rates
- **Architecture-Agnostic**: Works with any Transformer-based diffusion model

## Performance Analysis

AdaCorrection demonstrates significant quality improvements across various caching methods:

| Method | Baseline FID↓ | + AdaCorrection FID↓ | Improvement | FPS↑ | HR↑ |
|--------|---------------|----------------------|-------------|------|-----|
| FastCache | 4.46 | 4.37 | -0.09 | 15.5 | 83.5% |
| TeaCache | 5.09 | 4.54 | -0.55 | 15.7 | 77.9% |
| AdaCache | 4.75 | 4.43 | -0.32 | 14.8 | 78.1% |
| LazyDiT | 4.91 | 4.55 | -0.36 | 15.8 | 75.2% |
| FBCache | 4.48 | 4.38 | -0.10 | 14.7 | 82.1% |
| Full Recompute | 4.42 | - | - | 11.6 | 0% |

**Key Findings:**
- AdaCorrection consistently improves generation quality across all caching methods
- Quality improvements come with minimal computational overhead (FPS changes within ±0.4)
- Cache hit rates increase by 5-9 percentage points
- FastCache + AdaCorrection achieves the strongest performance (FID 4.37, FPS 15.7, HR 83.5%)

## Usage: Implementing AdaCorrection

### Basic Usage

```python
from xfuser.model_executor.cache.diffusers_adapters.flux import apply_cache_on_transformer
from diffusers import FluxPipeline

# Load your diffusion model
pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")

# Apply AdaCorrection with FastCache
apply_cache_on_transformer(
    pipeline.unet.transformer,
    use_cache="Fast",
    rel_l1_thresh=0.05,
    motion_threshold=0.1,
    enable_adacorrection=True,      # Enable AdaCorrection
    adacorr_gamma=1.0,              # Sensitivity parameter γ
    adacorr_lambda=1.0,             # Spatial weight λ
    return_hidden_states_first=False,
    num_steps=30
)

# Run inference with AdaCorrection
result = pipeline(
    prompt="a serene landscape with mountains and a lake",
    num_inference_steps=30,
)
```

### Advanced Configuration

```python
# Fine-tune AdaCorrection parameters
apply_cache_on_transformer(
    pipeline.unet.transformer,
    use_cache="Fast",
    enable_adacorrection=True,
    adacorr_gamma=1.0,      # Higher γ = more responsive to offsets
    adacorr_lambda=1.0,     # Higher λ = more weight on spatial variation
    # ... other parameters
)
```

### Parameter Sensitivity

- **γ (gamma)**: Controls sensitivity to offset magnitude
  - `γ = 0.5`: Under-correction (FID 4.56, HR 75.2%)
  - `γ = 1.0`: Optimal balance (FID 4.37, HR 83.5%) ⭐
  - `γ = 2.0`: Over-correction (FID 4.65, FPS 15.1)

- **λ (lambda)**: Controls spatial contribution
  - `λ = 0.5`: Reduces reuse (HR 72.3%)
  - `λ = 1.0`: Optimal balance ⭐
  - `λ = 2.0`: Slightly harms quality (FID 4.62)

**Recommended Settings**: `γ = 1.0` and `λ = 1.0` provide optimal balance between quality and efficiency.

## Integration with Other Caching Methods

AdaCorrection is designed to be plug-and-play with existing caching methods:

### With FastCache

```python
apply_cache_on_transformer(
    transformer,
    use_cache="Fast",
    enable_adacorrection=True,
    # FastCache parameters
    rel_l1_thresh=0.05,
    motion_threshold=0.1,
    # AdaCorrection parameters
    adacorr_gamma=1.0,
    adacorr_lambda=1.0,
)
```

### With TeaCache

```python
apply_cache_on_transformer(
    transformer,
    use_cache="Tea",
    enable_adacorrection=True,
    adacorr_gamma=1.0,
    adacorr_lambda=1.0,
)
```

### With AdaCache

```python
apply_cache_on_transformer(
    transformer,
    use_cache="Ada",
    enable_adacorrection=True,
    adacorr_gamma=1.0,
    adacorr_lambda=1.0,
)
```

## Theoretical Properties

### Error Propagation Bound

**Proposition 1 (Bounded Error Propagation)**: Assume each Transformer block `Block_ℓ` is L-Lipschitz and that the cached input is reused with lag `τ ≥ 0`. Under the adaptive interpolation `ĥ_{t,l+1} = (1 - λ_t^l) h̃_{t,l+1} + λ_t^l h_{t,l+1}`, the instantaneous deviation is bounded by:

```
||h_{t,l+1} - ĥ_{t,l+1}||_2 ≤ (1 - λ_t^l) * L * τ * S_t^l
```

This theoretical bound ensures that correction error is controlled and does not accumulate unboundedly.

### Asymptotic Convergence

**Theorem 1 (Convergence of Adaptive Cache Correction)**: Under mild conditions (bounded offset scores, vanishing variance, bounded reuse lag), the corrected hidden states converge to the true hidden states:

```
lim_{t→∞} ||h_t^l - ĥ_t^l||_2 = 0
```

This guarantees that AdaCorrection maintains long-term stability and quality.

## Benchmarking AdaCorrection

Compare AdaCorrection with other acceleration methods:

```bash
# Run benchmark with AdaCorrection
python benchmark/cache_execute.py \
    --model_type flux \
    --cache_methods None Fast "Fast+AdaCorrection" \
    --num_inference_steps 30 \
    --enable_adacorrection \
    --adacorr_gamma 1.0 \
    --adacorr_lambda 1.0
```

## Cross-Model Generalization

AdaCorrection provides consistent improvements across different model architectures and datasets:

| Backbone | Dataset | Baseline FID↓ | + AdaCorrection FID↓ | Improvement |
|----------|---------|---------------|----------------------|-------------|
| DiT-B/2 | FFHQ-256 | 6.13 | 5.97 | -0.16 |
| DiT-L/2 | LSUN-Church-256 | 5.72 | 5.55 | -0.17 |
| DiT-XL/2 | ImageNet-512 | 5.89 | 5.62 | -0.27 |
| PixArt-α | ImageNet-256 | 7.20 | 6.85 | -0.35 |
| SGDiff | COCO-512 | 8.65 | 8.10 | -0.55 |
| StableDiff. | FFHQ-1024 | 9.11 | 8.72 | -0.39 |

## Future Directions

- **Learned Correction Weights**: Training specialized correction weight functions for different model architectures
- **Hierarchical Correction**: Multi-level offset correction for even greater efficiency
- **Dynamic Parameter Adaptation**: Real-time adaptation of γ and λ based on content complexity
- **Cross-Modal Extension**: Extending AdaCorrection to video and 3D generation models

