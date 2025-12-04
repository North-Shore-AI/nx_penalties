# ADR-003: Elastic Net Regularizer

## Status

Proposed

## Context

Elastic Net combines L1 and L2 regularization, balancing sparsity induction (L1) with smooth shrinkage (L2). The combination often outperforms either alone, particularly when:

- Features are correlated (L2 handles grouping, L1 selects)
- Some sparsity is desired but not extreme
- Training stability is important (L2 smooths L1's sharp corners)

For Tinkex, Elastic Net provides a single regularizer that captures both behaviors without requiring users to manually compose L1 + L2 specs.

## Decision

Implement `Tinkex.Regularizer.ElasticNet` with configurable L1/L2 ratio.

### Interface

```elixir
defmodule Tinkex.Regularizer.ElasticNet do
  @behaviour Tinkex.Regularizer

  @doc """
  Elastic Net regularizer combining L1 and L2 penalties.

  Loss = alpha * L1 + (1 - alpha) * L2

  Where alpha=1 is pure L1, alpha=0 is pure L2.
  """

  @impl true
  def compute(_data, logprobs, opts \\ []) do
    alpha = Keyword.get(opts, :alpha, 0.5)
    target = Keyword.get(opts, :target, :logprobs)

    tensor = case target do
      :logprobs -> logprobs
      :probs -> Nx.exp(logprobs)
      {:field, key} -> extract_field(data, key)
    end

    # L1 component
    l1_value = Nx.sum(Nx.abs(tensor))

    # L2 component
    l2_value = Nx.sum(Nx.power(tensor, 2))

    # Combined: alpha * L1 + (1 - alpha) * L2
    alpha_tensor = Nx.tensor(alpha, type: Nx.type(tensor))
    one_minus_alpha = Nx.tensor(1.0 - alpha, type: Nx.type(tensor))

    elastic_value = Nx.add(
      Nx.multiply(alpha_tensor, l1_value),
      Nx.multiply(one_minus_alpha, l2_value)
    )

    {elastic_value, %{
      "elastic_net" => Nx.to_number(elastic_value),
      "l1_component" => Nx.to_number(l1_value),
      "l2_component" => Nx.to_number(l2_value),
      "alpha" => alpha
    }}
  end

  @impl true
  def name, do: "elastic_net"
end
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `:alpha` | float | `0.5` | L1/L2 mixing ratio (1.0 = pure L1, 0.0 = pure L2) |
| `:target` | atom \| tuple | `:logprobs` | What to regularize |
| `:reduce` | `:sum` \| `:mean` | `:sum` | Reduction method |

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
