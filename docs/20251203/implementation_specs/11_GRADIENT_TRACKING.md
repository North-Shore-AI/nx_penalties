# 11: Gradient Tracking Implementation Specification

## Overview

Gradient tracking provides visibility into which regularizers dominate the training signal. This is essential for debugging training instability, tuning hyperparameters, and understanding gradient flow.

**Origin**: Ported from `Tinkex.Regularizer.GradientTracker`.

**See also**: [ADR-009: Gradient Tracking](./regularizer_adrs/adrs/ADR-009_gradient_tracking.md)

## Module: Tinkex.Regularizer.GradientTracker

### File Location
```
tinkex/regularizer/gradient_tracker.ex
```

### Module Structure

```elixir
defmodule Tinkex.Regularizer.GradientTracker do
  @moduledoc """
  Computes gradient norms for regularizers using Nx automatic differentiation.

  ## Purpose

  When training with multiple regularizers, it's crucial to understand which
  regularizers contribute most to the gradient signal. This module provides:

  - Per-regularizer gradient norms
  - Total composed gradient norm
  - Gradient ratio analysis

  ## Usage

  Enable in pipeline computation:

      {total, metrics} = Tinkex.Regularizer.Pipeline.compute(data, logprobs, base_loss_fn, regularizers: regularizers, track_grad_norms: true)

      output.regularizers["l1_grad_norm"]      # L2 norm of L1 regularizer's gradient
      output.regularizers["total_grad_norm"]   # Combined gradient norm

  ## Performance Note

  Gradient tracking requires additional backward passes. Only enable when
  debugging or for periodic monitoring (e.g., every 100 steps).

  ## Important: What Are We Differentiating?

  These functions compute ∂regularizer/∂(pipeline_input), NOT ∂regularizer/∂params.

  The "pipeline input" is whatever tensor you pass to `Pipeline.compute/3`—
  typically model outputs, activations, or logprobs. This tells you how
  sensitive each regularizer is to changes in that tensor.

  True parameter-wise gradient tracking (∂loss/∂params) would require
  hooking into the training step itself, which is outside  Tinkex.Regularizer.Pipeline' scope.
  """

  require Logger
end
```

---

## Function: compute_grad_norm/2

### Specification

```elixir
@doc """
Compute L2 norm of gradients from a loss function.

## Parameters

- `loss_fn` - Function `(tensor) -> scalar_tensor`
- `tensor` - Input to differentiate with respect to

## Returns

Float representing L2 norm: `||∇f||₂ = sqrt(Σ grad²)`

## Examples

    iex> loss_fn = fn x -> Nx.sum(Nx.abs(x)) end
    iex> GradientTracker.compute_grad_norm(loss_fn, Nx.tensor([1.0, -2.0, 3.0]))
    1.732...  # sqrt(3) since ∂|x|/∂x = sign(x) = [1, -1, 1]

    iex> loss_fn = fn x -> Nx.sum(Nx.power(x, 2)) end
    iex> GradientTracker.compute_grad_norm(loss_fn, Nx.tensor([1.0, 2.0, 3.0]))
    7.483...  # sqrt(4 + 16 + 36) since ∂x²/∂x = 2x
"""
@spec compute_grad_norm(
        loss_fn :: (Nx.Tensor.t() -> Nx.Tensor.t()),
        tensor :: Nx.Tensor.t()
      ) :: float() | nil
def compute_grad_norm(loss_fn, tensor) do
  try do
    # Nx.Defn.grad/1 returns a function that computes the gradient
    # The loss_fn must be defn-compatible (only Nx ops, no side effects)
    grad_fn = Nx.Defn.grad(loss_fn)
    grad_tensor = grad_fn.(tensor)

    grad_tensor
    |> Nx.flatten()
    |> Nx.power(2)
    |> Nx.sum()
    |> Nx.sqrt()
    |> Nx.to_number()
  rescue
    e ->
      Logger.warning("""
      Gradient computation failed: #{inspect(e)}

      This usually means the loss function contains non-differentiable operations.
      Consider disabling gradient tracking or fixing the regularizer function.
      """)
      nil
  end
end
```

### Implementation Notes

1. **Nx.Defn.grad/1**: Unlike PyTorch's `retain_graph=True`, Nx computes gradients symbolically. No graph retention needed. The function returns a new function that computes the gradient.

2. **Error Handling**: Some operations may not be differentiable. Return `nil` with warning rather than crashing. The pipeline integration includes a `*_grad_norm_error` metric when this occurs.

3. **Flattening**: Flatten gradient tensor before computing norm to handle arbitrary shapes.

---

## Function: pipeline_grad_norms/2

### Specification

```elixir
@doc """
Compute gradient norms for all regularizers in a pipeline.

## Parameters

- `pipeline` - Pipeline struct with regularizer entries
- `tensor` - Input tensor to differentiate with respect to

## Returns

Map of `%{"name_grad_norm" => float, ...}`

## Examples

    pipeline =  Tinkex.Regularizer.Pipeline.pipeline([
      {:l1, weight: 0.001},
      {:l2, weight: 0.01}
    ])

    norms = GradientTracker.pipeline_grad_norms(pipeline, tensor)
    # %{"l1_grad_norm" => 1.732, "l2_grad_norm" => 7.483}
"""
@spec pipeline_grad_norms(Pipeline.t(), Nx.Tensor.t()) :: map()
def pipeline_grad_norms(pipeline, tensor) do
  pipeline.entries
  |> Enum.filter(fn {_, _, _, _, enabled} -> enabled end)
  |> Enum.flat_map(fn {name, regularizer_fn, _weight, opts, _enabled} ->
    loss_fn = fn t -> regularizer_fn.(t, opts) end
    case compute_grad_norm(loss_fn, tensor) do
      nil ->
        # Include error indicator in metrics
        [{"#{name}_grad_norm", nil}, {"#{name}_grad_norm_error", true}]
      norm ->
        [{"#{name}_grad_norm", norm}]
    end
  end)
  |> Map.new()
end
```

---

## Function: total_grad_norm/2

### Specification

```elixir
@doc """
Compute gradient norm for the total weighted pipeline loss.

## Formula

    total = Σ(weight_i × regularizer_i(tensor))
    result = ||∇_tensor total||₂

## Parameters

- `pipeline` - Pipeline struct
- `tensor` - Input tensor

## Returns

Float representing total gradient norm.

## Examples

    norm = GradientTracker.total_grad_norm(pipeline, tensor)
"""
@spec total_grad_norm(Pipeline.t(), Nx.Tensor.t()) :: float() | nil
def total_grad_norm(pipeline, tensor) do
  total_loss_fn = fn t ->
    pipeline.entries
    |> Enum.filter(fn {_, _, _, _, enabled} -> enabled end)
    |> Enum.map(fn {_name, regularizer_fn, weight, opts, _enabled} ->
      Nx.multiply(regularizer_fn.(t, opts), weight)
    end)
    |> Enum.reduce(Nx.tensor(0.0), &Nx.add/2)
  end

  compute_grad_norm(total_loss_fn, tensor)
end
```

---

## Pipeline Integration

### API Relationship

The top-level ` Tinkex.Regularizer.Pipeline.compute/3` delegates to ` Tinkex.Regularizer.Pipeline.Pipeline.compute/3`:

```elixir
# Top-level convenience API (in  Tinkex.Regularizer.Pipeline module)
defmodule  Tinkex.Regularizer.Pipeline do
  @doc """
  Compute pipeline regularizers with optional gradient tracking.

  Delegates to ` Tinkex.Regularizer.Pipeline.Pipeline.compute/3`.

  ## Options

    * `:track_grad_norms` - Compute gradient norms (default: `false`)
    * `:extra_args` - Additional args merged into each regularizer's opts

  ## Examples

      {total, metrics} =  Tinkex.Regularizer.Pipeline.compute(pipeline, tensor)
      {total, metrics} = Tinkex.Regularizer.Pipeline.compute(data, logprobs, base_loss_fn, regularizers: regularizers, track_grad_norms: true)
  """
  def compute(pipeline, tensor, opts \\ []) do
     Tinkex.Regularizer.Pipeline.Pipeline.compute(pipeline, tensor, opts)
  end
end
```

### Enhanced Pipeline.compute/3

```elixir
defmodule  Tinkex.Regularizer.Pipeline.Pipeline do
  def compute(%__MODULE__{} = pipeline, tensor, opts \\ []) do
    track_grad_norms = Keyword.get(opts, :track_grad_norms, false)
    extra_args = Keyword.get(opts, :extra_args, [])

    # Compute regularizers (existing logic from 04_PIPELINE.md)
    {total, base_metrics} = compute_regularizers(pipeline, tensor, extra_args)

    # Optionally add gradient metrics
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

  # Private: compute regularizers without gradient tracking
  defp compute_regularizers(pipeline, tensor, extra_args) do
    # Implementation as specified in 04_PIPELINE.md
    # Uses extra_args merged into each regularizer's opts
  end
end
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `:track_grad_norms` | boolean | `false` | Compute gradient norms for each regularizer |

---

## Test Cases

```elixir
describe "compute_grad_norm/2" do
  test "computes correct L2 norm for L1 regularizer gradient" do
    # Gradient of sum(|x|) is sign(x)
    loss_fn = fn x -> Nx.sum(Nx.abs(x)) end
    tensor = Nx.tensor([1.0, -2.0, 3.0])

    norm = GradientTracker.compute_grad_norm(loss_fn, tensor)

    # sign([1, -2, 3]) = [1, -1, 1], L2 norm = sqrt(3)
    assert_close(norm, :math.sqrt(3))
  end

  test "computes correct L2 norm for L2 regularizer gradient" do
    # Gradient of sum(x²) is 2x
    loss_fn = fn x -> Nx.sum(Nx.power(x, 2)) end
    tensor = Nx.tensor([1.0, 2.0, 3.0])

    norm = GradientTracker.compute_grad_norm(loss_fn, tensor)

    # 2*[1, 2, 3] = [2, 4, 6], L2 norm = sqrt(4 + 16 + 36) = sqrt(56)
    assert_close(norm, :math.sqrt(56))
  end

  test "handles non-differentiable operations gracefully" do
    # Nx.argmax is not differentiable
    loss_fn = fn x -> Nx.argmax(x) end
    tensor = Nx.tensor([1.0, 2.0, 3.0])

    # Should return nil instead of crashing
    norm = GradientTracker.compute_grad_norm(loss_fn, tensor)
    assert norm == nil
  end

  test "handles multidimensional tensors" do
    loss_fn = fn x -> Nx.sum(Nx.abs(x)) end
    tensor = Nx.tensor([[1.0, -2.0], [-3.0, 4.0]])

    norm = GradientTracker.compute_grad_norm(loss_fn, tensor)

    # sign of each element, L2 norm = sqrt(4) = 2
    assert_close(norm, 2.0)
  end
end

describe "pipeline_grad_norms/2" do
  test "computes norms for all pipeline entries" do
    pipeline =  Tinkex.Regularizer.Pipeline.pipeline([
      {:l1, weight: 0.001},
      {:l2, weight: 0.01}
    ])
    tensor = Nx.tensor([1.0, 2.0, 3.0])

    norms = GradientTracker.pipeline_grad_norms(pipeline, tensor)

    assert Map.has_key?(norms, "l1_grad_norm")
    assert Map.has_key?(norms, "l2_grad_norm")
    assert norms["l1_grad_norm"] > 0
    assert norms["l2_grad_norm"] > 0
  end
end

describe "total_grad_norm/2" do
  test "combines weighted gradients correctly" do
    pipeline =  Tinkex.Regularizer.Pipeline.pipeline([
      {:l1, weight: 1.0}
    ])
    tensor = Nx.tensor([1.0, 1.0, 1.0])

    norm = GradientTracker.total_grad_norm(pipeline, tensor)

    # L1 gradient is sign(x) = [1, 1, 1], L2 norm = sqrt(3)
    assert_close(norm, :math.sqrt(3))
  end
end

describe "pipeline integration" do
  test "compute returns gradient metrics when tracking enabled" do
    pipeline =  Tinkex.Regularizer.Pipeline.pipeline([
      {:l1, weight: 0.001},
      {:l2, weight: 0.01}
    ])
    tensor = Nx.tensor([1.0, 2.0, 3.0])

    {_total, metrics} = Tinkex.Regularizer.Pipeline.compute(data, logprobs, base_loss_fn, regularizers: regularizers, track_grad_norms: true)

    assert Map.has_key?(metrics, "l1_grad_norm")
    assert Map.has_key?(metrics, "l2_grad_norm")
    assert Map.has_key?(metrics, "total_grad_norm")
  end

  test "compute does not include gradient metrics when tracking disabled" do
    pipeline =  Tinkex.Regularizer.Pipeline.pipeline([{:l1, weight: 0.001}])
    tensor = Nx.tensor([1.0, 2.0, 3.0])

    {_total, metrics} =  Tinkex.Regularizer.Pipeline.compute(pipeline, tensor)

    refute Map.has_key?(metrics, "l1_grad_norm")
    refute Map.has_key?(metrics, "total_grad_norm")
  end
end
```

---

## Performance Considerations

### Cost Analysis

| Pipeline Size | Mode | Forward Passes | Backward Passes |
|--------------|------|----------------|-----------------|
| N regularizers | No tracking | N | 0 |
| N regularizers | With tracking | N | N + 1 |

### Recommended Usage Patterns

```elixir
# Pattern 1: Periodic tracking (recommended)
def train_step(pipeline, batch, step) do
  track = rem(step, 100) == 0
  {total, metrics} =  Tinkex.Regularizer.Pipeline.compute(pipeline, batch, track_grad_norms: track)
  # ...
end

# Pattern 2: Debug mode only
def train_step(pipeline, batch, debug: debug) do
  {total, metrics} =  Tinkex.Regularizer.Pipeline.compute(pipeline, batch, track_grad_norms: debug)
  # ...
end

# Pattern 3: Conditional on instability detection
def train_step(pipeline, batch, prev_loss) do
  # First compute without gradient tracking
  {total, _} =  Tinkex.Regularizer.Pipeline.compute(pipeline, batch)

  # Check for loss spike
  loss_spike = prev_loss != nil and Nx.to_number(total) / prev_loss > 2.0

  # Re-compute with gradient tracking only if spike detected
  if loss_spike do
    {total, metrics} =  Tinkex.Regularizer.Pipeline.compute(pipeline, batch, track_grad_norms: true)
    {total, metrics, Nx.to_number(total)}
  else
    {total, %{}, Nx.to_number(total)}
  end
end
```

---

## Telemetry Events

| Event | Measurements | Metadata |
|-------|-------------|----------|
| `[:nx_regularizers, :gradient, :computed]` | `%{total_norm: float}` | `%{pipeline_size: int}` |
| `[:nx_regularizers, :gradient, :failed]` | `%{duration: ns}` | `%{reason: term}` |

---

## Integration Checklist

- [ ] `compute_grad_norm/2` with error handling
- [ ] `pipeline_grad_norms/2` for all entries
- [ ] `total_grad_norm/2` for composed loss
- [ ] Pipeline `compute/3` integration with `:track_grad_norms` option
- [ ] Test coverage for edge cases
- [ ] Telemetry integration
- [ ] Documentation with examples
