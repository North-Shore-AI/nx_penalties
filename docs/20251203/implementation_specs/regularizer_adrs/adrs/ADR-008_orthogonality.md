# ADR-008: Orthogonality Regularizer

## Status

Proposed

## Context

Orthogonality regularization encourages learned representations to be orthogonal (uncorrelated). This is particularly relevant for LoRA fine-tuning because:

1. **LoRA structure** - LoRA adds low-rank matrices A and B where the update is BA. Orthogonality in these matrices improves conditioning.

2. **Representation diversity** - orthogonal features capture distinct information, reducing redundancy.

3. **Training stability** - orthogonal weight matrices have condition number 1, optimal for gradient flow.

4. **Capacity utilization** - prevents multiple dimensions from learning the same feature.

Since Tinker doesn't expose LoRA weights directly, we apply orthogonality constraints to output representations (logprobs or attention patterns if available).

## Decision

Implement orthogonality via the tensor primitive `NxPenalties.Constraints.orthogonality/2`, with a Tinkex adapter to fit the data/logprobs regularizer contract.

### Interface

```elixir
# NxPenalties primitive (tensor-only)
penalty = NxPenalties.Constraints.orthogonality(logprobs,
  mode: :soft,      # or :hard/:spectral
  axis: :sequence,  # or :vocabulary/:rows
  normalize: true
)

# Tinkex adapter (data-aware signature)
defmodule Tinkex.Regularizers.Orthogonality do
  @behaviour Tinkex.Regularizer

  @impl true
  def compute(_data, logprobs, opts \\ []) do
    penalty = NxPenalties.Constraints.orthogonality(logprobs, opts)
    {penalty, %{"orthogonality" => Nx.to_number(penalty)}}
  end
end
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `:mode` | atom | `:soft` | Orthogonality measure (`:soft`, `:hard`, `:spectral`) |
| `:axis` | atom | `:sequence` | Which axis to orthogonalize (`:sequence`, `:vocabulary`) |
| `:normalize` | boolean | `true` | Normalize rows before Gram computation |

## Consequences

### Positive

- Encourages diverse, non-redundant representations
- Improves gradient flow through orthogonal structure
- Can prevent mode collapse in certain settings
- Interpretable metrics (correlation coefficients)

### Negative

- Requires computing Gram matrix (O(n²) in sequence length)
- May conflict with desired correlations in output
- Not directly applicable to LoRA weights (API limitation)
- Spectral mode requires SVD (expensive or unavailable)

### Neutral

- Multiple modes for different tradeoffs
- Axis selection allows flexibility

## Implementation Notes

### Computational Complexity

| Mode | Complexity | Notes |
|------|------------|-------|
| `:soft` | O(n² × d) | n = positions, d = vocab |
| `:hard` | O(n² × d) | Same as soft |
| `:spectral` | O(n³) | Full SVD, often approximated |

For typical LLM outputs (seq=2048, vocab=32k), soft mode computes a 2048×2048 correlation matrix - manageable but not free.

### Relationship to LoRA

LoRA updates have the form: W' = W + BA where B ∈ R^{d×r}, A ∈ R^{r×k}.

Ideally, we'd regularize:
- B^T B ≈ I (columns of B orthogonal)
- A A^T ≈ I (rows of A orthogonal)

Without weight access, we use output orthogonality as a proxy, which encourages the combined effect to produce diverse outputs.

### Mode Selection Guide

| Mode | Behavior | Use Case |
|------|----------|----------|
| `:soft` | Only penalizes correlations | Allow diagonal variation |
| `:hard` | Enforce M@M^T = I | Strict orthogonality |
| `:spectral` | Uniform singular values | Isotropic representations |

### Typical Weight Range

| Weight | Effect |
|--------|--------|
| 0.001 | Light diversity encouragement |
| 0.01 | Moderate orthogonality pressure |
| 0.1 | Strong, may affect task performance |

## Alternatives Considered

### 1. Covariance Regularization
Penalize covariance instead of correlation. Doesn't account for scale differences.

### 2. Mutual Information Minimization
Minimize I(X_i; X_j) between dimensions. Requires density estimation, complex.

### 3. Decorrelation via Batch Normalization
Whitening transformation in forward pass. Not applicable to Tinker's API.

### 4. SRIP (Spectral Restricted Isometry Property)
Advanced orthogonality for deep networks. Too complex for initial implementation.

## References

- Bansal et al. (2018). "Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?"
- Saxe et al. (2013). "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks"
- Arjovsky et al. (2016). "Unitary Evolution Recurrent Neural Networks"
- Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"
