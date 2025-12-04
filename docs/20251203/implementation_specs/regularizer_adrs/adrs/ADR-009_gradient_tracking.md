# ADR-009: Gradient Tracking for Regularizers

## Status

Proposed

## Context

When training with multiple regularizers, it's crucial to understand which penalties dominate the gradient signal. Without this visibility:

1. **Debugging is hard** - Training instability could come from any regularizer
2. **Hyperparameter tuning is blind** - No way to know if weights are balanced
3. **Gradient explosion is opaque** - Can't identify the culprit regularizer

Tinkex currently implements `GradientTracker` for this purpose. This ADR proposes porting this functionality to NxPenalties as part of the pipeline infrastructure.

**Note**: Pipeline entries use a 5-tuple `{name, fn, weight, opts, enabled}`. Some examples below show the 4-tuple form for brevity; the `enabled` boolean is implicit in these cases.

## Decision

Implement `NxPenalties.GradientTracker` for computing L2 gradient norms of individual penalties and total composed loss.

### Interface

```elixir
defmodule NxPenalties.GradientTracker do
  @moduledoc """
  Computes gradient norms for regularizers using Nx automatic differentiation.

  Useful for monitoring which penalties dominate the training signal and
  detecting gradient explosion or vanishing gradients.

  ## Usage

  Enable gradient tracking in pipeline computation:

      pipeline = NxPenalties.pipeline([
        {:l1, weight: 0.001},
        {:l2, weight: 0.01}
      ])

      {total, metrics} = NxPenalties.compute(pipeline, tensor, track_grad_norms: true)

      metrics["l1_grad_norm"]       # L2 norm of L1 penalty's gradient
      metrics["l2_grad_norm"]       # L2 norm of L2 penalty's gradient
      metrics["total_grad_norm"]    # L2 norm of combined gradient

  ## Implementation Note

  Uses `Nx.Defn.grad/1` for automatic differentiation. Unlike PyTorch's
  `retain_graph=True`, Nx computes gradients symbolically without graph retention.

  **Important**: The `loss_fn` must be defn-compatible (only Nx ops, no side effects).
  """

  import Nx.Defn

  @doc """
  Compute L2 norm of gradients from a loss function.

  ## Parameters

  - `loss_fn` - Function `(tensor) -> scalar_tensor` (must be defn-compatible)
  - `tensor` - Input to differentiate with respect to

  ## Returns

  Float representing L2 norm: `sqrt(sum(grad²))`, or `nil` if differentiation fails.

  ## Examples

      iex> loss_fn = fn x -> Nx.sum(Nx.abs(x)) end
      iex> GradientTracker.compute_grad_norm(loss_fn, Nx.tensor([1.0, -2.0, 3.0]))
      1.732...  # sqrt(3) since gradient of |x| is sign(x)
  """
  @spec compute_grad_norm(
          loss_fn :: (Nx.Tensor.t() -> Nx.Tensor.t()),
          tensor :: Nx.Tensor.t()
        ) :: float() | nil
  def compute_grad_norm(loss_fn, tensor) do
    # Nx.Defn.grad/1 returns a function that computes the gradient
    grad_fn = Nx.Defn.grad(loss_fn)
    grad_tensor = grad_fn.(tensor)

    grad_tensor
    |> Nx.flatten()
    |> Nx.power(2)
    |> Nx.sum()
    |> Nx.sqrt()
    |> Nx.to_number()
  rescue
    _ -> nil
  end

  @doc """
  Compute gradient norms for all penalties in a pipeline.

  ## Parameters

  - `pipeline` - Pipeline struct with penalty entries
  - `tensor` - Input tensor to differentiate with respect to

  ## Returns

  Map of `%{"penalty_name_grad_norm" => float, ...}`

  ## Examples

      norms = GradientTracker.pipeline_grad_norms(pipeline, tensor)
      # %{"l1_grad_norm" => 1.732, "l2_grad_norm" => 5.0, ...}
  """
  @spec pipeline_grad_norms(Pipeline.t(), Nx.Tensor.t()) :: map()
  def pipeline_grad_norms(pipeline, tensor) do
    pipeline.entries
    |> Enum.filter(fn {_, _, _, _, enabled} -> enabled end)
    |> Enum.flat_map(fn {name, penalty_fn, _weight, opts, _enabled} ->
      loss_fn = fn t -> penalty_fn.(t, opts) end
      case compute_grad_norm(loss_fn, tensor) do
        nil -> [{:"#{name}_grad_norm", nil}, {:"#{name}_grad_norm_error", true}]
        norm -> [{:"#{name}_grad_norm", norm}]
      end
    end)
    |> Map.new()
  end

  @doc """
  Compute gradient norm for the total weighted pipeline loss.

  ## Parameters

  - `pipeline` - Pipeline struct
  - `tensor` - Input tensor

  ## Returns

  Float representing total gradient norm.

  ## Formula

      total = Σ(weight_i × penalty_i(tensor))
      result = ||∇_tensor total||_2
  """
  @spec total_grad_norm(Pipeline.t(), Nx.Tensor.t()) :: float() | nil
  def total_grad_norm(pipeline, tensor) do
    total_loss_fn = fn t ->
      pipeline.entries
      |> Enum.filter(fn {_, _, _, _, enabled} -> enabled end)
      |> Enum.map(fn {_name, penalty_fn, weight, opts, _enabled} ->
        Nx.multiply(penalty_fn.(t, opts), weight)
      end)
      |> Enum.reduce(Nx.tensor(0.0), &Nx.add/2)
    end

    compute_grad_norm(total_loss_fn, tensor)
  end
end
```

### Pipeline Integration

```elixir
defmodule NxPenalties.Pipeline do
  # Existing compute/3 enhanced with gradient tracking

  def compute(pipeline, tensor, opts \\ []) do
    track_grad_norms = Keyword.get(opts, :track_grad_norms, false)

    # ... existing penalty computation ...

    metrics = if track_grad_norms do
      grad_metrics = GradientTracker.pipeline_grad_norms(pipeline, tensor)
      total_norm = GradientTracker.total_grad_norm(pipeline, tensor)

      base_metrics
      |> Map.merge(grad_metrics)
      |> Map.put("total_grad_norm", total_norm)
    else
      base_metrics
    end

    {total, metrics}
  end
end
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `:track_grad_norms` | boolean | `false` | Compute gradient norms for each penalty |

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
# Pattern: Track every 100 steps
track = rem(step, 100) == 0
{total, metrics} = NxPenalties.compute(pipeline, tensor, track_grad_norms: track)
```

### Gradient Ratio Analysis

Common pattern for debugging:

```elixir
metrics = compute_with_tracking(pipeline, tensor)

# Ratio of each penalty's gradient to total
Enum.each(pipeline.entries, fn {name, _, _, _} ->
  ratio = metrics["#{name}_grad_norm"] / metrics["total_grad_norm"]
  IO.puts("#{name}: #{Float.round(ratio * 100, 1)}% of gradient")
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
# Emit gradient norm telemetry
:telemetry.execute(
  [:nx_penalties, :gradient, :computed],
  %{
    total_norm: total_norm,
    max_penalty_norm: max_norm,
    min_penalty_norm: min_norm
  },
  %{pipeline_size: length(pipeline.entries)}
)
```

## Typical Use Cases

### 1. Debugging Training Instability

```elixir
{total, metrics} = NxPenalties.compute(pipeline, tensor, track_grad_norms: true)

if metrics["total_grad_norm"] > 100.0 do
  Logger.warning("Gradient explosion detected!")
  # Find culprit
  culprit = Enum.max_by(pipeline.entries, fn {name, _, _, _} ->
    metrics["#{name}_grad_norm"]
  end)
  Logger.warning("Largest contributor: #{elem(culprit, 0)}")
end
```

### 2. Weight Balancing

```elixir
# Ensure no single penalty dominates
norms = for {name, _, weight, _} <- pipeline.entries do
  weighted_norm = metrics["#{name}_grad_norm"] * weight
  {name, weighted_norm}
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
def log_gradient_stats(pipeline, tensor, step) when rem(step, 100) == 0 do
  {_, metrics} = NxPenalties.compute(pipeline, tensor, track_grad_norms: true)

  :telemetry.execute([:training, :gradient_stats], %{
    step: step,
    total_norm: metrics["total_grad_norm"],
    norms: Map.filter(metrics, fn {k, _} -> String.ends_with?(k, "_grad_norm") end)
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
