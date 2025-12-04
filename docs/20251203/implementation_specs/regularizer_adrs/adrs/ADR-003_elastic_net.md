# ADR-003: Elastic Net Regularizer

## Status

Proposed

## Context

Elastic Net combines L1 and L2 regularization, balancing sparsity induction (L1) with smooth shrinkage (L2). The combination often outperforms either alone, particularly when:

- Features are correlated (L2 handles grouping, L1 selects)
- Some sparsity is desired but not extreme
- Training stability is important (L2 smooths L1's sharp corners)

Elastic Net provides a single regularizer that captures both behaviors without requiring users to manually compose L1 + L2 specs. The tensor primitive lives in `NxPenalties.Penalties.elastic_net/2`; Tinkex supplies the data-aware adapter.

## Decision

Implement `NxPenalties.Penalties.elastic_net/2` as the numeric primitive with configurable L1/L2 ratio, and a Tinkex adapter that resolves the target tensor.

### Interface

```elixir
# Tensor primitive (NxPenalties)
elastic_value = NxPenalties.Penalties.elastic_net(tensor,
  l1_ratio: 0.5, # 1.0 = L1, 0.0 = L2
  lambda: 1.0,
  reduction: :sum # or :mean
)

# Tinkex adapter (data-aware)
defmodule Tinkex.Regularizers.ElasticNet do
  @behaviour Tinkex.Regularizer

  @impl true
  def compute(data, logprobs, opts \\ []) do
    target = Keyword.get(opts, :target, :logprobs)
    l1_ratio = Keyword.get(opts, :l1_ratio, 0.5)
    reduction = Keyword.get(opts, :reduction, :sum)

    tensor =
      case target do
        :logprobs -> logprobs
        :probs -> Nx.exp(logprobs)
        {:field, key} -> fetch_field!(data, key)
      end

    elastic_value =
      NxPenalties.Penalties.elastic_net(tensor,
        l1_ratio: l1_ratio,
        reduction: reduction
      )

    {elastic_value, %{
      "elastic_net" => Nx.to_number(elastic_value),
      "l1_ratio" => l1_ratio
    }}
  end
end
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `:l1_ratio` | float | `0.5` | L1/L2 mixing ratio (1.0 = pure L1, 0.0 = pure L2) |
| `:target` | atom \| tuple | `:logprobs` | What to regularize (Tinkex adapter) |
| `:reduction` | `:sum` \| `:mean` | `:sum` | Reduction method |
| `:lambda` | number | `1.0` | Scaling inside the NxPenalties primitive (optional) |

## Consequences

### Positive

- Single regularizer provides both L1 and L2 benefits
- Tunable alpha allows finding optimal sparsity/smoothness tradeoff
- More stable than pure L1 due to L2 component
- Returns both components in metrics for analysis

### Negative

- Additional hyperparameter (alpha) to tune
- Slightly higher compute than L1 or L2 alone (but negligible)
- Users may not understand L1/L2 tradeoffs

### Neutral

- alpha=0.5 is reasonable default for most cases
- Metrics expose components for debugging

## Implementation Notes

### Alpha Selection Guidelines

| Alpha | Behavior | Use Case |
|-------|----------|----------|
| 0.9-1.0 | Mostly L1 | When sparsity is primary goal |
| 0.5 | Balanced | General purpose, default |
| 0.1-0.3 | Mostly L2 | When smoothness matters, some sparsity desired |
| 0.0 | Pure L2 | Use L2 regularizer directly instead |

### Gradient Properties

Gradient is weighted combination of L1 and L2 gradients:
- L1 gradient: sign(x)
- L2 gradient: 2x
- Elastic: alpha * sign(x) + (1-alpha) * 2x

The L2 component ensures gradient is never exactly zero (except at origin), improving optimization.

### Relationship to Pipeline Composition

Users can achieve similar results by composing L1 + L2 in a pipeline:

```elixir
# Equivalent to ElasticNet with alpha=0.5, total weight 0.02
specs = [
  %RegularizerSpec{fn: &L1.compute/3, weight: 0.01, name: "l1"},
  %RegularizerSpec{fn: &L2.compute/3, weight: 0.01, name: "l2"}
]
```

ElasticNet is a convenience when you want a single weight controlling total regularization strength, with alpha controlling the mix.

## Alternatives Considered

### 1. Separate L1 and L2 only
Rejected - Elastic Net is common enough to warrant first-class support.

### 2. Group Elastic Net
More complex variant for structured sparsity. Could be future ADR if needed.

### 3. Adaptive Elastic Net
Alpha that changes during training. Would require stateful regularizer, out of scope.

## References

- Zou & Hastie (2005). "Regularization and Variable Selection via the Elastic Net"
- scikit-learn ElasticNet implementation
