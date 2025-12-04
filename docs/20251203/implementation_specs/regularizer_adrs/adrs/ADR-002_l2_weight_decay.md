# ADR-002: L2 Weight Decay Regularizer

## Status

Proposed

## Context

L2 regularization (Ridge/Tikhonov) adds a penalty proportional to the squared magnitude of values. Unlike L1, it doesn't drive values to zero but shrinks them proportionally, preferring many small values over few large ones.

In Tinker's LoRA context, L2 can be applied to:

1. **Logprobs** - penalizes extreme confidence (very negative logprobs for rejected tokens)
2. **Probability distribution** - encourages uniform-ish outputs
3. **Custom loss intermediates** - when provided via Datum

Note: Traditional "weight decay" is applied by the optimizer (Adam in Tinker's case). This regularizer operates on forward pass outputs, not weights directly.

## Decision

Implement L2 weight decay as a tensor primitive in `NxPenalties.Penalties.l2/2`, with a Tinkex adapter to choose targets and centering.

### Interface

```elixir
# Tensor primitive (NxPenalties)
l2_value = NxPenalties.Penalties.l2(tensor,
  lambda: 1.0,
  reduction: :sum, # or :mean
  center: nil      # or :mean / number
)

# Tinkex adapter (data-aware)
defmodule Tinkex.Regularizers.L2 do
  @behaviour Tinkex.Regularizer

  @impl true
  def compute(data, logprobs, opts \\ []) do
    target = Keyword.get(opts, :target, :logprobs)
    reduction = Keyword.get(opts, :reduction, :sum)
    center = Keyword.get(opts, :center, nil)

    tensor =
      case target do
        :logprobs -> logprobs
        :probs -> Nx.exp(logprobs)
        {:field, key} -> fetch_field!(data, key)
      end

    l2_value = NxPenalties.Penalties.l2(tensor, reduction: reduction, center: center)

    squared = Nx.pow(tensor, 2)

    {l2_value, %{
      "l2_raw" => Nx.to_number(Nx.sum(squared)),
      "l2_mean" => Nx.to_number(Nx.mean(squared)),
      "l2_max" => Nx.to_number(Nx.reduce_max(squared))
    }}
  end
end
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `:target` | atom \| tuple | `:logprobs` | What to regularize (Tinkex adapter) |
| `:center` | nil \| :mean \| number | `nil` | Center before squaring (Tinkex adapter + NxPenalties) |
| `:reduction` | `:sum` \| `:mean` | `:sum` | Reduction method (both) |
| `:lambda` | number | `1.0` | Scaling inside the NxPenalties primitive (optional) |

## Consequences

### Positive

- Smooth, differentiable everywhere (unlike L1)
- Well-behaved gradients for stable training
- Prevents extreme values without forcing sparsity
- Centering option allows penalizing deviation from reference

### Negative

- Doesn't induce sparsity (may want L1 + L2 combo)
- Operating on logprobs vs weights is conceptually different from traditional L2
- Squaring can amplify numerical issues with very large values

### Neutral

- Returns multiple metrics (raw, mean, max) for monitoring
- Centering is optional but useful for specific use cases

## Implementation Notes

### Numerical Stability

For logprobs in range [-inf, 0], squaring produces [0, inf]. Very negative logprobs (rejected tokens) will dominate. Consider:

```elixir
# Clip extreme values before squaring
clipped = Nx.clip(logprobs, -100, 0)
squared = Nx.power(clipped, 2)
```

### Gradient Flow

L2 has clean gradients: d/dx(x²) = 2x. No special handling needed for GradientTracker.

### Relationship to AdamW

Tinker's optimizer supports weight decay in AdamParams. This regularizer is complementary - it operates on outputs, not parameters. Both can be used together.

### Typical Weight Range

For logprob-based L2, typical weights are 1e-5 to 1e-3. Higher weights aggressively smooth the output distribution.

## Alternatives Considered

### 1. Frobenius norm on logprob matrix
Equivalent to L2 for 2D case. Could add as alias.

### 2. Spectral norm regularization
More complex, requires SVD. Could be separate ADR if needed.

### 3. L2 with momentum (historical penalty)
Would require state across training steps. Out of scope for stateless regularizer.

## References

- Hoerl & Kennard (1970). "Ridge Regression"
- Loshchilov & Hutter (2017). "Decoupled Weight Decay Regularization" (AdamW)
