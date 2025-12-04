# ADR-001: L1 Sparsity Regularizer

## Status

Proposed

## Context

L1 regularization (Lasso) adds a penalty proportional to the absolute value of model outputs or parameters. It encourages sparsity by driving small values to exactly zero.

In the context of LoRA fine-tuning with Tinker, we operate on logprobs (log probabilities) returned from the forward pass rather than direct weight access. The L1 penalty can be applied to:

1. **Logprobs** - encourages the model to be decisive (few high-probability tokens)
2. **Logprob deltas** - encourages minimal change from uniform distribution
3. **Loss function inputs** - when custom loss functions provide intermediate values

## Decision

Implement L1 sparsity as a tensor primitive in `NxPenalties.Penalties.l1/2` and a thin Tinkex adapter for data-aware target selection.

### Interface

```elixir
# Tensor primitive (NxPenalties)
l1_value = NxPenalties.Penalties.l1(tensor,
  lambda: 1.0,
  reduction: :sum  # or :mean
)

# Tinkex adapter (data-aware)
defmodule Tinkex.Regularizers.L1 do
  @behaviour Tinkex.Regularizer

  @impl true
  def compute(data, logprobs, opts \\ []) do
    target = Keyword.get(opts, :target, :logprobs)
    reduction = Keyword.get(opts, :reduction, :sum)

    tensor =
      case target do
        :logprobs -> logprobs
        :probs -> Nx.exp(logprobs)
        {:field, key} -> fetch_field!(data, key)
      end

    l1_value = NxPenalties.Penalties.l1(tensor, reduction: reduction)

    {l1_value, %{
      "l1_raw" => Nx.to_number(Nx.sum(Nx.abs(tensor))),
      "l1_mean" => Nx.to_number(Nx.mean(Nx.abs(tensor)))
    }}
  end
end
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `:target` | atom \| tuple | `:logprobs` | What to regularize (Tinkex adapter) |
| `:reduction` | `:sum` \| `:mean` | `:sum` | Reduction method (both) |
| `:lambda` | number | `1.0` | Scaling inside the NxPenalties primitive (optional) |

## Consequences

### Positive

- Simple, well-understood regularizer
- Encourages sparse, interpretable outputs
- Low computational overhead (single Nx operation)
- Composable with other regularizers via pipeline

### Negative

- L1 on logprobs may not be meaningful for all use cases
- No direct weight access in Tinker API limits traditional L1 usage
- May need tuning of weight parameter for different model scales

### Neutral

- Returns both sum and mean metrics for flexibility in monitoring

## Implementation Notes

### Gradient Considerations

L1 has a non-differentiable point at zero. Nx handles this via subgradient (returning 0 at the discontinuity). For `track_grad_norms: true`, the GradientTracker will use the same subgradient.

### Numerical Stability

Logprobs are typically in range [-inf, 0]. Taking absolute value is stable. No special handling needed.

### Typical Weight Range

For logprob-based L1, typical weights are 1e-4 to 1e-2 depending on vocabulary size and sequence length.

## Alternatives Considered

### 1. L1 on weight deltas via API extension
Rejected - would require Tinker API changes to expose LoRA weight values.

### 2. L1 on probability distribution (exp(logprobs))
Included as option via `:target => :probs`. More interpretable but less stable numerically.

### 3. Soft L1 (Huber-style)
Could be added as separate regularizer if sharp corners of L1 cause training instability.

## References

- Tibshirani, R. (1996). "Regression Shrinkage and Selection via the Lasso"
- [Nx.abs/1 documentation](https://hexdocs.pm/nx/Nx.html#abs/1)
