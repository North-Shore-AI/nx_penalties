# ADR-009: Gradient Tracking for Regularizers

## Status

Proposed

## Context

When training with multiple regularizers, it's crucial to understand which penalties dominate the gradient signal. Without this visibility:

1. **Debugging is hard** - Training instability could come from any regularizer
2. **Hyperparameter tuning is blind** - No way to know if weights are balanced
3. **Gradient explosion is opaque** - Can't identify the culprit regularizer

Tinkex currently implements `GradientTracker` for this purpose. NxPenalties remains tensor-only; Tinkex owns the data-aware pipeline and gradient tracking for regularizers.

**Note**: Tinkex configures regularizers via `RegularizerSpec` (function + weight + name). NxPenalties' built-in pipeline is single-tensor only and does not handle data-aware gradient tracking.

## Decision

Implement gradient tracking inside Tinkex's data-aware regularizer pipeline (NxPenalties remains tensor-only).

### Interface

```elixir
# Tinkex-side gradient tracking (data-aware pipeline)
defmodule Tinkex.Regularizer.GradientTracker do
  import Nx.Defn

  @doc """
  Compute L2 norm of gradients from a defn-compatible loss_fn.
  """
  def compute_grad_norm(loss_fn, tensor) do
    grad_fn = Nx.Defn.grad(loss_fn)
    grad_fn.(tensor)
    |> Nx.flatten()
    |> Nx.pow(2)
    |> Nx.sum()
    |> Nx.sqrt()
    |> Nx.to_number()
  rescue
    _ -> nil
  end

  @doc """
  Compute per-regularizer gradient norms in a data-aware pipeline.
  """
  def pipeline_grad_norms(regularizers, data, logprobs) do
    regularizers
    |> Enum.filter(& &1.enabled)
    |> Enum.flat_map(fn spec ->
      loss_fn = fn lp ->
        {val, _} = Tinkex.Regularizer.execute(spec.fn, data, lp, spec.opts)
        val
      end

      case compute_grad_norm(loss_fn, logprobs) do
        nil -> [{:"#{spec.name}_grad_norm", nil}, {:"#{spec.name}_grad_norm_error", true}]
        norm -> [{:"#{spec.name}_grad_norm", norm}]
      end
    end)
    |> Map.new()
  end

  @doc """
  Compute total gradient norm across all regularizers (weighted sum).
  """
  def total_grad_norm(regularizers, data, logprobs) do
    loss_fn = fn lp ->
      regularizers
      |> Enum.filter(& &1.enabled)
      |> Enum.reduce(Nx.tensor(0.0), fn spec, acc ->
        {val, _} = Tinkex.Regularizer.execute(spec.fn, data, lp, spec.opts)
        Nx.add(acc, Nx.multiply(val, spec.weight))
      end)
    end

    compute_grad_norm(loss_fn, logprobs)
  rescue
    _ -> nil
  end
end
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `:track_grad_norms` | boolean | `false` | Compute gradient norms for each regularizer in the Tinkex pipeline |

## Consequences

### Positive

- **Training visibility** - See which penalties contribute to gradients
- **Debugging** - Identify sources of gradient explosion/vanishing
- **Hyperparameter guidance** - Know if weight scaling is balanced
- **Parity with tinkex** - Feature already proven useful

### Negative

- **Performance overhead** - Requires extra backward passes
- **Not always needed** - Many users won't need this
- **Complexity** - Adds another dimension to pipeline output

### Neutral

- Optional feature, disabled by default
- Only computed when explicitly requested

## Implementation Notes

### Performance Considerations

Gradient tracking requires computing gradients for each penalty separately. This approximately doubles the cost per penalty. For a pipeline with N penalties:

| Mode | Forward Passes | Backward Passes |
|------|---------------|-----------------|
| No tracking | N | 0 |
| With tracking | N | N + 1 |

**Recommendation**: Only enable for debugging or periodic monitoring, not every training step.

```elixir
# Pattern: Track every 100 steps (Tinkex pipeline)
track = rem(step, 100) == 0
{:ok, output} = Tinkex.Regularizer.Pipeline.compute(data, logprobs, base_loss_fn,
  regularizers: regularizers,
  track_grad_norms: track
)
metrics = output.regularizers
```

### Gradient Ratio Analysis

Common pattern for debugging:

```elixir
{:ok, output} = Tinkex.Regularizer.Pipeline.compute(data, logprobs, base_loss_fn,
  regularizers: regularizers,
  track_grad_norms: true
)
metrics = output.regularizers

# Ratio of each penalty's gradient to total
Enum.each(regularizers, fn spec ->
  ratio = metrics["#{spec.name}_grad_norm"] / metrics["total_grad_norm"]
  IO.puts("#{spec.name}: #{Float.round(ratio * 100, 1)}% of gradient")
end)
```

### Handling Non-Differentiable Operations

Some operations may not be differentiable. The tracker should handle this gracefully:

```elixir
def compute_grad_norm(loss_fn, tensor) do
  try do
    # ... compute gradient ...
  rescue
    e ->
      Logger.warning("Gradient computation failed: #{inspect(e)}")
      nil  # Return nil; pipeline will add *_grad_norm_error metric
  end
end
```

### Telemetry Integration

```elixir
# Emit gradient norm telemetry (Tinkex)
:telemetry.execute(
  [:tinkex, :regularizer, :gradients],
  %{total_norm: total_norm},
  %{regularizer_count: length(regularizers)}
)
```

## Typical Use Cases

### 1. Debugging Training Instability

```elixir
{:ok, output} = Tinkex.Regularizer.Pipeline.compute(data, logprobs, base_loss_fn,
  regularizers: regularizers,
  track_grad_norms: true
)
metrics = output.regularizers

if metrics["total_grad_norm"] > 100.0 do
  Logger.warning("Gradient explosion detected!")
  # Find culprit
  culprit = Enum.max_by(regularizers, fn spec ->
    metrics["#{spec.name}_grad_norm"]
  end)
  Logger.warning("Largest contributor: #{culprit.name}")
end
```

### 2. Weight Balancing

```elixir
# Ensure no single regularizer dominates
norms = for spec <- regularizers do
  weighted_norm = metrics["#{spec.name}_grad_norm"] * spec.weight
  {spec.name, weighted_norm}
end

max_norm = Enum.max_by(norms, &elem(&1, 1)) |> elem(1)
min_norm = Enum.min_by(norms, &elem(&1, 1)) |> elem(1)

if max_norm / min_norm > 100 do
  Logger.warning("Gradient imbalance: consider adjusting weights")
end
```

### 3. Monitoring Dashboard

```elixir
# Periodic logging for training dashboard
def log_gradient_stats(data, logprobs, regularizers, base_loss_fn, step) when rem(step, 100) == 0 do
  {:ok, output} =
    Tinkex.Regularizer.Pipeline.compute(data, logprobs, base_loss_fn,
      regularizers: regularizers,
      track_grad_norms: true
    )

  :telemetry.execute([:training, :gradient_stats], %{
    step: step,
    total_norm: output.regularizers["total_grad_norm"],
    norms: Map.filter(output.regularizers, fn {k, _} -> String.ends_with?(k, "_grad_norm") end)
  })
end
```

## Alternatives Considered

### 1. Always compute gradient norms

Rejected - too expensive for production training.

### 2. Use hooks/callbacks instead of explicit tracking

Rejected - Nx doesn't support PyTorch-style hooks. Explicit computation is cleaner.

### 3. Sample-based gradient estimation

Could use finite differences for approximate gradients. Rejected - Nx autodiff is available and exact.

## References

- Tinkex `GradientTracker` implementation (original source)
- Nx autodiff documentation
- "Gradient Flow in Recurrent Nets" (Hochreiter & Schmidhuber, 1997) - gradient norm monitoring for RNNs
