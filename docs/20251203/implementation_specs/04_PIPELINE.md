# 04: Pipeline Implementation Specification

## Overview

The Pipeline module provides composition infrastructure for combining multiple penalties into a single training objective. This is the "glue" that makes individual penalty functions useful in practice.

## Design Goals

1. **Declarative** - Define objectives as data, not imperative code
2. **Composable** - Add/remove/reweight penalties easily
3. **Observable** - Return metrics alongside total loss
4. **JIT-Compatible** - Compile to efficient computation graph

## Module: NxPenalties.Pipeline

### File Location
```
lib/nx_penalties/pipeline.ex
```

### Core Struct

```elixir
defmodule NxPenalties.Pipeline do
  @moduledoc """
  Compose multiple penalties into a single objective function.

  ## Building a Pipeline

      pipeline =
        NxPenalties.Pipeline.new()
        |> NxPenalties.Pipeline.add(:l1, &NxPenalties.Penalties.l1/2,
             weight: 0.001, opts: [lambda: 1.0])
        |> NxPenalties.Pipeline.add(:l2, &NxPenalties.Penalties.l2/2,
             weight: 0.01)
        |> NxPenalties.Pipeline.add(:entropy, &NxPenalties.Divergences.entropy/2,
             weight: 0.1, opts: [mode: :bonus])

  ## Computing Penalties

      {total_loss, metrics} = NxPenalties.Pipeline.compute(pipeline, tensor)

      # metrics => %{
      #   "l1" => 0.023,
      #   "l1_weighted" => 0.000023,
      #   "l2" => 1.45,
      #   "l2_weighted" => 0.0145,
      #   "entropy" => -2.3,
      #   "entropy_weighted" => -0.23,
      #   "total" => -0.215477
      # }

  ## Dynamic Weights

  Weights can be tensors for curriculum learning:

      schedule_weight = Nx.tensor(current_epoch / max_epochs)
      pipeline = NxPenalties.Pipeline.update_weight(pipeline, :kl, schedule_weight)
  """

  defstruct [
    :entries,      # List of {name, fn, weight, opts, enabled}
    :reduction,    # How to combine: :sum | :mean
    :scale        # Global scale factor
  ]

  @type entry :: {atom(), function(), number() | Nx.Tensor.t(), keyword(), boolean()}

  @type t :: %__MODULE__{
    entries: [entry()],
    reduction: :sum | :mean,
    scale: number() | Nx.Tensor.t()
  }
end
```

### Builder Functions

```elixir
@doc """
Create a new empty pipeline.

## Options

  * `:reduction` - How to combine penalties. Default: `:sum`
  * `:scale` - Global scale factor. Default: `1.0`
"""
@spec new(keyword()) :: t()
def new(opts \\ []) do
  %__MODULE__{
    entries: [],
    reduction: Keyword.get(opts, :reduction, :sum),
    scale: Keyword.get(opts, :scale, 1.0)
  }
end

@doc """
Add a penalty to the pipeline.

## Parameters

  * `pipeline` - The pipeline to add to
  * `name` - Atom name for this penalty (used in metrics)
  * `penalty_fn` - Function with signature `(tensor, opts) -> scalar_tensor`
  * `opts` - Configuration options:
    * `:weight` - Multiplier for this penalty. Default: `1.0`
    * `:opts` - Options passed to penalty function. Default: `[]`
    * `:enabled` - Whether penalty is active. Default: `true`

## Examples

      pipeline
      |> add(:l1, &NxPenalties.Penalties.l1/2, weight: 0.01)
      |> add(:kl, &NxPenalties.Divergences.kl_divergence/3,
           weight: 0.1, opts: [reduction: :mean])
"""
@spec add(t(), atom(), function(), keyword()) :: t()
def add(%__MODULE__{} = pipeline, name, penalty_fn, opts \\ []) do
  weight = Keyword.get(opts, :weight, 1.0)
  penalty_opts = Keyword.get(opts, :opts, [])
  enabled = Keyword.get(opts, :enabled, true)

  entry = {name, penalty_fn, weight, penalty_opts, enabled}

  %{pipeline | entries: pipeline.entries ++ [entry]}
end

@doc """
Remove a penalty from the pipeline by name.
"""
@spec remove(t(), atom()) :: t()
def remove(%__MODULE__{} = pipeline, name) do
  entries = Enum.reject(pipeline.entries, fn {n, _, _, _, _} -> n == name end)
  %{pipeline | entries: entries}
end

@doc """
Update the weight of an existing penalty.

Useful for curriculum learning or dynamic adjustment.
"""
@spec update_weight(t(), atom(), number() | Nx.Tensor.t()) :: t()
def update_weight(%__MODULE__{} = pipeline, name, new_weight) do
  entries = Enum.map(pipeline.entries, fn
    {^name, fn_, _weight, opts, enabled} -> {name, fn_, new_weight, opts, enabled}
    entry -> entry
  end)
  %{pipeline | entries: entries}
end

@doc """
Enable or disable a penalty by name.
"""
@spec set_enabled(t(), atom(), boolean()) :: t()
def set_enabled(%__MODULE__{} = pipeline, name, enabled) do
  entries = Enum.map(pipeline.entries, fn
    {^name, fn_, weight, opts, _} -> {name, fn_, weight, opts, enabled}
    entry -> entry
  end)
  %{pipeline | entries: entries}
end
```

### Computation Functions

```elixir
@doc """
Compute all penalties and return total + metrics.

## Parameters

  * `pipeline` - The configured pipeline
  * `tensor` - Primary input tensor (e.g., model outputs or weights)
  * `extra_args` - Additional arguments for penalties that need them
    (e.g., reference distribution for KL)

## Returns

  `{total_penalty, metrics_map}`

  Where `metrics_map` contains:
  - `"{name}"` - Raw penalty value
  - `"{name}_weighted"` - Weight-adjusted value
  - `"total"` - Sum of all weighted penalties

  For gradient norm tracking (debugging which regularizer dominates training),
  see [11_GRADIENT_TRACKING.md](./11_GRADIENT_TRACKING.md).
"""
@spec compute(t(), Nx.Tensor.t(), keyword()) :: {Nx.Tensor.t(), map()}
def compute(%__MODULE__{} = pipeline, tensor, extra_args \\ [])

# When no entries, return zero
def compute(%__MODULE__{entries: []}, _tensor, _extra_args) do
  {Nx.tensor(0.0), %{"total" => 0.0}}
end

def compute(%__MODULE__{} = pipeline, tensor, extra_args) do
  # Compute each enabled penalty
  results =
    pipeline.entries
    |> Enum.filter(fn {_, _, _, _, enabled} -> enabled end)
    |> Enum.map(fn {name, penalty_fn, weight, opts, _enabled} ->
      # Merge extra_args into opts
      full_opts = Keyword.merge(opts, extra_args)

      # Call penalty function
      raw_value = apply_penalty(penalty_fn, tensor, full_opts)
      weighted_value = Nx.multiply(raw_value, weight)

      {name, raw_value, weighted_value}
    end)

  # Build metrics map
  metrics =
    results
    |> Enum.flat_map(fn {name, raw, weighted} ->
      [
        {Atom.to_string(name), Nx.to_number(raw)},
        {"#{name}_weighted", Nx.to_number(weighted)}
      ]
    end)
    |> Map.new()

  # Compute total
  total =
    results
    |> Enum.map(fn {_, _, weighted} -> weighted end)
    |> Enum.reduce(Nx.tensor(0.0), &Nx.add/2)
    |> Nx.multiply(pipeline.scale)

  metrics = Map.put(metrics, "total", Nx.to_number(total))

  {total, metrics}
end

# Helper to apply penalty with varying arities
defp apply_penalty(penalty_fn, tensor, opts) when is_function(penalty_fn, 1) do
  penalty_fn.(tensor)
end

defp apply_penalty(penalty_fn, tensor, opts) when is_function(penalty_fn, 2) do
  penalty_fn.(tensor, opts)
end
```

### Defn-Compatible Computation

For use inside JIT-compiled training loops:

```elixir
defmodule NxPenalties.Pipeline.Defn do
  @moduledoc """
  Defn-compatible pipeline computation for JIT compilation.

  Since pipelines are built dynamically at runtime, we compile them
  to a single defn function for execution efficiency.
  """

  import Nx.Defn

  @doc """
  Compile a pipeline to a defn function.

  Returns a function `(tensor, opts) -> {total, individual_values}`.

  ## Example

      compiled_fn = NxPenalties.Pipeline.Defn.compile(pipeline)
      {total, values} = compiled_fn.(tensor, [])
  """
  def compile(%NxPenalties.Pipeline{} = pipeline) do
    # Extract penalty functions and weights
    entries = Enum.filter(pipeline.entries, fn {_, _, _, _, enabled} -> enabled end)

    fn tensor, opts ->
      compute_compiled(tensor, entries, pipeline.scale, opts)
    end
  end

  defnp compute_compiled(tensor, entries, scale, opts) do
    # This would be generated code based on pipeline structure
    # For now, show the pattern:

    # Example with 2 penalties:
    # p1 = penalty_fn_1.(tensor, opts)
    # p2 = penalty_fn_2.(tensor, opts)
    # total = w1 * p1 + w2 * p2

    # Actual implementation would use macro to generate this
    Nx.tensor(0.0)
  end
end
```

### Macro-Based Compilation (Advanced)

```elixir
defmodule NxPenalties.Pipeline.Compiler do
  @moduledoc """
  Compile pipelines to optimized defn code at compile time.

  ## Usage

      defmodule MyTraining do
        use NxPenalties.Pipeline.Compiler

        @pipeline NxPenalties.Pipeline.new()
          |> NxPenalties.Pipeline.add(:l1, &NxPenalties.Penalties.l1/2, weight: 0.01)
          |> NxPenalties.Pipeline.add(:l2, &NxPenalties.Penalties.l2/2, weight: 0.001)

        # Generates: defn compute_penalties(tensor, opts)
        compile_pipeline(@pipeline)
      end
  """

  defmacro compile_pipeline(pipeline) do
    quote do
      import Nx.Defn

      defn compute_penalties(tensor, opts \\ []) do
        # Generated code based on pipeline structure
        unquote(generate_penalty_code(pipeline))
      end
    end
  end

  defp generate_penalty_code(pipeline) do
    # At compile time, inspect pipeline and generate AST
    # This is advanced metaprogramming territory
    quote do
      Nx.tensor(0.0)  # Placeholder
    end
  end
end
```

---

## Multi-Input Pipelines

Some penalties need multiple inputs (e.g., KL divergence needs P and Q).

```elixir
defmodule NxPenalties.Pipeline.Multi do
  @moduledoc """
  Pipelines with multiple named inputs.

  ## Example

      pipeline =
        NxPenalties.Pipeline.Multi.new()
        |> NxPenalties.Pipeline.Multi.add(:kl,
             fn inputs, _opts ->
               NxPenalties.Divergences.kl_divergence(
                 inputs.model_logprobs,
                 inputs.reference_logprobs
               )
             end,
             weight: 0.1)

      inputs = %{
        model_logprobs: model_output,
        reference_logprobs: base_model_output
      }

      {total, metrics} = NxPenalties.Pipeline.Multi.compute(pipeline, inputs)
  """

  defstruct [:entries, :reduction, :scale]

  @type t :: %__MODULE__{
    entries: [{atom(), function(), term(), keyword(), boolean()}],
    reduction: :sum | :mean,
    scale: term()
  }

  def new(opts \\ []), do: # similar to Pipeline.new

  def add(pipeline, name, penalty_fn, opts \\ []), do: # similar

  @doc """
  Compute with named inputs map.
  """
  @spec compute(t(), map(), keyword()) :: {Nx.Tensor.t(), map()}
  def compute(%__MODULE__{} = pipeline, inputs, extra_args \\ []) when is_map(inputs) do
    results =
      pipeline.entries
      |> Enum.filter(fn {_, _, _, _, enabled} -> enabled end)
      |> Enum.map(fn {name, penalty_fn, weight, opts, _enabled} ->
        raw_value = penalty_fn.(inputs, Keyword.merge(opts, extra_args))
        weighted_value = Nx.multiply(raw_value, weight)
        {name, raw_value, weighted_value}
      end)

    # Same aggregation as Pipeline.compute
    # ...
  end
end
```

---

## Telemetry Integration

```elixir
defmodule NxPenalties.Pipeline.Telemetry do
  @moduledoc """
  Telemetry instrumentation for pipeline computation.
  """

  @doc """
  Wrap pipeline compute with telemetry events.
  """
  def with_telemetry(%NxPenalties.Pipeline{} = pipeline, tensor, opts \\ []) do
    metadata = %{
      pipeline_size: length(pipeline.entries),
      tensor_shape: Nx.shape(tensor)
    }

    :telemetry.span(
      [:nx_penalties, :pipeline, :compute],
      metadata,
      fn ->
        {total, metrics} = NxPenalties.Pipeline.compute(pipeline, tensor, opts)
        {{total, metrics}, Map.merge(metadata, %{metrics: metrics})}
      end
    )
  end
end
```

---

## Usage Examples

### Basic Training Loss

```elixir
defmodule MyTraining do
  import Nx.Defn

  def build_pipeline do
    NxPenalties.Pipeline.new()
    |> NxPenalties.Pipeline.add(:l2, &NxPenalties.Penalties.l2/2,
         weight: 0.001, opts: [lambda: 1.0])
    |> NxPenalties.Pipeline.add(:entropy, &NxPenalties.Divergences.entropy/2,
         weight: 0.01, opts: [mode: :penalty])
  end

  def training_step(pipeline, params, x, y) do
    {y_pred, _state} = model_forward(params, x)

    # Task loss
    task_loss = Axon.Losses.mean_squared_error(y, y_pred)

    # Regularization
    {reg_loss, metrics} = NxPenalties.Pipeline.compute(pipeline, y_pred)

    total_loss = Nx.add(task_loss, reg_loss)

    {total_loss, Map.put(metrics, "task_loss", Nx.to_number(task_loss))}
  end
end
```

### Curriculum Learning

```elixir
def update_pipeline_for_epoch(pipeline, epoch, max_epochs) do
  # Increase KL weight over training
  kl_weight = epoch / max_epochs * 0.5  # 0 -> 0.5

  # Decrease entropy bonus over training
  entropy_weight = (1.0 - epoch / max_epochs) * 0.1  # 0.1 -> 0

  pipeline
  |> NxPenalties.Pipeline.update_weight(:kl, kl_weight)
  |> NxPenalties.Pipeline.update_weight(:entropy, entropy_weight)
end
```

### RL Policy Loss

```elixir
def build_ppo_pipeline do
  NxPenalties.Pipeline.Multi.new()
  |> NxPenalties.Pipeline.Multi.add(:policy_loss,
       fn inputs, _opts ->
         # Clipped surrogate objective
         ratio = Nx.exp(Nx.subtract(inputs.new_log_probs, inputs.old_log_probs))
         clipped = Nx.clip(ratio, 1.0 - 0.2, 1.0 + 0.2)
         Nx.negate(Nx.mean(Nx.min(
           Nx.multiply(ratio, inputs.advantages),
           Nx.multiply(clipped, inputs.advantages)
         )))
       end,
       weight: 1.0)
  |> NxPenalties.Pipeline.Multi.add(:entropy_bonus,
       fn inputs, _opts ->
         NxPenalties.Divergences.entropy(inputs.new_log_probs, mode: :bonus)
       end,
       weight: 0.01)
  |> NxPenalties.Pipeline.Multi.add(:kl_penalty,
       fn inputs, _opts ->
         NxPenalties.Divergences.kl_divergence(
           inputs.new_log_probs,
           inputs.old_log_probs
         )
       end,
       weight: 0.1)
end
```

---

## Test Cases

```elixir
describe "Pipeline" do
  test "empty pipeline returns zero" do
    pipeline = NxPenalties.Pipeline.new()
    tensor = Nx.tensor([1.0, 2.0, 3.0])
    {total, metrics} = NxPenalties.Pipeline.compute(pipeline, tensor)
    assert_close(total, Nx.tensor(0.0))
    assert metrics["total"] == 0.0
  end

  test "single penalty" do
    pipeline =
      NxPenalties.Pipeline.new()
      |> NxPenalties.Pipeline.add(:l1, &NxPenalties.Penalties.l1/2, weight: 0.1)

    tensor = Nx.tensor([1.0, 2.0, 3.0])
    {total, metrics} = NxPenalties.Pipeline.compute(pipeline, tensor)

    assert Map.has_key?(metrics, "l1")
    assert Map.has_key?(metrics, "l1_weighted")
    assert_close(metrics["l1_weighted"], Nx.to_number(total))
  end

  test "multiple penalties sum correctly" do
    pipeline =
      NxPenalties.Pipeline.new()
      |> NxPenalties.Pipeline.add(:l1, &NxPenalties.Penalties.l1/2, weight: 1.0)
      |> NxPenalties.Pipeline.add(:l2, &NxPenalties.Penalties.l2/2, weight: 1.0)

    tensor = Nx.tensor([1.0, 2.0])
    {total, metrics} = NxPenalties.Pipeline.compute(pipeline, tensor)

    expected = metrics["l1_weighted"] + metrics["l2_weighted"]
    assert_close(metrics["total"], expected)
  end

  test "disabled penalties are skipped" do
    pipeline =
      NxPenalties.Pipeline.new()
      |> NxPenalties.Pipeline.add(:l1, &NxPenalties.Penalties.l1/2, weight: 1.0)
      |> NxPenalties.Pipeline.add(:l2, &NxPenalties.Penalties.l2/2, weight: 1.0, enabled: false)

    tensor = Nx.tensor([1.0, 2.0])
    {_total, metrics} = NxPenalties.Pipeline.compute(pipeline, tensor)

    assert Map.has_key?(metrics, "l1")
    refute Map.has_key?(metrics, "l2")
  end

  test "weight update" do
    pipeline =
      NxPenalties.Pipeline.new()
      |> NxPenalties.Pipeline.add(:l1, &NxPenalties.Penalties.l1/2, weight: 0.1)

    updated = NxPenalties.Pipeline.update_weight(pipeline, :l1, 0.5)

    tensor = Nx.tensor([1.0])
    {_, metrics1} = NxPenalties.Pipeline.compute(pipeline, tensor)
    {_, metrics2} = NxPenalties.Pipeline.compute(updated, tensor)

    assert metrics2["l1_weighted"] == metrics1["l1_weighted"] * 5
  end
end
```

---

## Integration Checklist

- [ ] Pipeline struct is well-documented
- [ ] Builder functions have clear semantics
- [ ] Compute returns both total and individual metrics
- [ ] Disabled penalties are correctly skipped
- [ ] Weight updates work correctly
- [ ] Multi-input variant handles named tensors
- [ ] Telemetry wrapper provided
- [ ] Usage examples for common scenarios
- [ ] Curriculum learning pattern documented
