# 05: Axon Integration Implementation Specification

## Overview

This document specifies the integration layer between NxPenalties and Axon. Since Axon explicitly rejects model-level regularization ("regularization is a concern of training/optimization and not the model"), we provide training loop helpers rather than layer modifications.

## Design Constraints

1. **No Axon Core Changes** - Work with existing Axon API
2. **Axon.Loop Compatible** - Integrate with standard training loops
3. **Minimal Boilerplate** - Make regularization easy to add
4. **State Threading** - Handle the functional state problem

## Module: NxPenalties.Integration.Axon

### File Location
```
lib/nx_penalties/integration/axon.ex
```

### Module Structure
```elixir
defmodule NxPenalties.Integration.Axon do
  @moduledoc """
  Helpers for integrating NxPenalties with Axon training loops.

  ## Integration Patterns

  ### Pattern 1: Wrap Loss Function

  The simplest approach - wrap your loss function with penalties:

      loss_fn = NxPenalties.Integration.Axon.wrap_loss(
        &Axon.Losses.mean_squared_error/2,
        &NxPenalties.Penalties.l2/2,
        lambda: 0.01
      )

      Axon.Loop.trainer(model, loss_fn, optimizer)

  ### Pattern 2: Custom Train Step

  For more control, build a custom training step:

      train_step = NxPenalties.Integration.Axon.build_train_step(
        model,
        &Axon.Losses.categorical_cross_entropy/2,
        pipeline,
        optimizer
      )

      Axon.Loop.loop(train_step)

  ### Pattern 3: Activity Regularization (Advanced)

  For regularizing intermediate activations:

      model =
        input
        |> Axon.dense(128)
        |> NxPenalties.Integration.Axon.capture_activation(:hidden1)
        |> Axon.relu()
        |> Axon.dense(10)

      # In training step, extract captured activations and apply penalties
  """
end
```

---

## Pattern 1: Loss Wrapping

### Function: wrap_loss/3

```elixir
@doc """
Wrap a loss function to include penalty terms.

The simplest integration pattern. Takes a base loss function and adds
penalty computation on the predictions.

## Parameters

  * `base_loss_fn` - Original loss function `(y_true, y_pred) -> scalar`
  * `penalty_fn` - Penalty function `(tensor, opts) -> scalar`
  * `opts` - Options:
    * `:lambda` - Weight for penalty term. Default: `0.01`
    * `:penalty_opts` - Options passed to penalty function

## Returns

A new loss function with signature `(y_true, y_pred) -> scalar`

## Example

    wrapped_loss = NxPenalties.Integration.Axon.wrap_loss(
      &Axon.Losses.mean_squared_error/2,
      &NxPenalties.Penalties.l2/2,
      lambda: 0.01
    )

    model
    |> Axon.Loop.trainer(wrapped_loss, optimizer)
    |> Axon.Loop.run(data, %{}, epochs: 10)
"""
@spec wrap_loss(function(), function(), keyword()) :: function()
def wrap_loss(base_loss_fn, penalty_fn, opts \\ []) do
  lambda = Keyword.get(opts, :lambda, 0.01)
  penalty_opts = Keyword.get(opts, :penalty_opts, [])

  fn y_true, y_pred ->
    base_loss = base_loss_fn.(y_true, y_pred)
    penalty = penalty_fn.(y_pred, penalty_opts)
    Nx.add(base_loss, Nx.multiply(penalty, lambda))
  end
end
```

### Function: wrap_loss_with_pipeline/3

```elixir
@doc """
Wrap a loss function with a full penalty pipeline.

More flexible than `wrap_loss/3` - supports multiple penalties
with individual weights.

## Parameters

  * `base_loss_fn` - Original loss function
  * `pipeline` - `NxPenalties.Pipeline` struct
  * `opts` - Additional options

## Returns

A new loss function. Note: metrics from pipeline are not accessible
with this pattern. Use `build_train_step/4` for metrics.

## Example

    pipeline =
      NxPenalties.Pipeline.new()
      |> NxPenalties.Pipeline.add(:l1, &NxPenalties.Penalties.l1/2, weight: 0.001)
      |> NxPenalties.Pipeline.add(:entropy, &NxPenalties.Divergences.entropy/2,
           weight: 0.01, opts: [mode: :penalty])

    wrapped_loss = NxPenalties.Integration.Axon.wrap_loss_with_pipeline(
      &Axon.Losses.categorical_cross_entropy/2,
      pipeline
    )
"""
@spec wrap_loss_with_pipeline(function(), NxPenalties.Pipeline.t(), keyword()) :: function()
def wrap_loss_with_pipeline(base_loss_fn, pipeline, opts \\ []) do
  fn y_true, y_pred ->
    base_loss = base_loss_fn.(y_true, y_pred)
    {penalty_total, _metrics} = NxPenalties.Pipeline.compute(pipeline, y_pred, opts)
    Nx.add(base_loss, penalty_total)
  end
end
```

---

## Pattern 2: Custom Train Step

### Function: build_train_step/4

```elixir
@doc """
Build a complete training step function with penalty support.

This pattern provides full control and access to metrics from
each penalty computation.

## Parameters

  * `model` - Axon model
  * `base_loss_fn` - Loss function `(y_true, y_pred) -> scalar`
  * `pipeline` - `NxPenalties.Pipeline` struct
  * `optimizer` - Polaris optimizer tuple

## Returns

A step function compatible with `Axon.Loop.loop/3`.

## Metrics

The step function adds these metrics to the loop state:
- `"base_loss"` - Task loss before penalties
- `"penalty_total"` - Sum of all penalties
- `"loss"` - Total loss (base + penalties)
- Plus individual penalty metrics from pipeline

## Example

    pipeline =
      NxPenalties.Pipeline.new()
      |> NxPenalties.Pipeline.add(:l2, &NxPenalties.Penalties.l2/2, weight: 0.01)

    train_step = NxPenalties.Integration.Axon.build_train_step(
      model,
      &Axon.Losses.mean_squared_error/2,
      pipeline,
      Polaris.Optimizers.adam(learning_rate: 0.001)
    )

    loop =
      Axon.Loop.loop(train_step)
      |> Axon.Loop.metric(:loss, "total_loss")
      |> Axon.Loop.metric(:base_loss, "base_loss")

    Axon.Loop.run(loop, data, %{}, epochs: 10)
"""
@spec build_train_step(Axon.t(), function(), NxPenalties.Pipeline.t(), term()) :: function()
def build_train_step(model, base_loss_fn, pipeline, optimizer) do
  {init_fn, predict_fn} = Axon.build(model, mode: :train)
  {opt_init_fn, opt_update_fn} = optimizer

  # Initialize function
  init = fn data, init_state ->
    params = init_fn.(data, init_state)
    opt_state = opt_init_fn.(params)
    %{
      model_state: params,
      optimizer_state: opt_state
    }
  end

  # Step function
  step = fn state, batch ->
    %{model_state: params, optimizer_state: opt_state} = state
    {x, y_true} = batch

    # Compute gradients
    {loss, grads} = Nx.Defn.value_and_grad(params, fn p ->
      y_pred = predict_fn.(p, x)

      # Base loss
      base_loss = base_loss_fn.(y_true, y_pred)

      # Penalties
      {penalty_total, _metrics} = NxPenalties.Pipeline.compute(pipeline, y_pred)

      # Total loss for gradient
      Nx.add(base_loss, penalty_total)
    end)

    # Also compute metrics (without grad)
    y_pred = predict_fn.(params, x)
    base_loss = base_loss_fn.(y_true, y_pred)
    {penalty_total, penalty_metrics} = NxPenalties.Pipeline.compute(pipeline, y_pred)

    # Update parameters
    {updates, new_opt_state} = opt_update_fn.(grads, opt_state, params)
    new_params = Polaris.Updates.apply_updates(params, updates)

    # Build metrics
    metrics = Map.merge(penalty_metrics, %{
      "base_loss" => Nx.to_number(base_loss),
      "penalty_total" => Nx.to_number(penalty_total),
      "loss" => Nx.to_number(loss)
    })

    new_state = %{
      model_state: new_params,
      optimizer_state: new_opt_state
    }

    {new_state, metrics}
  end

  {init, step}
end
```

---

## Pattern 3: Activity Regularization

### Challenge

Axon layers don't expose intermediate activations to the loss function. We need to capture them during the forward pass.

### Solution: Capture Layers

```elixir
@doc """
Insert a capture layer that stores activations in model state.

The captured activations can be extracted after the forward pass
and used for activity regularization.

## Parameters

  * `model` - Axon model (at desired capture point)
  * `name` - Atom name for this capture point

## Returns

Modified Axon model with capture layer inserted.

## Example

    model =
      Axon.input("input")
      |> Axon.dense(256)
      |> Axon.relu()
      |> NxPenalties.Integration.Axon.capture_activation(:layer1)
      |> Axon.dense(128)
      |> Axon.relu()
      |> NxPenalties.Integration.Axon.capture_activation(:layer2)
      |> Axon.dense(10)
      |> Axon.softmax()

    # In training, model state will contain captured activations
"""
@spec capture_activation(Axon.t(), atom()) :: Axon.t()
def capture_activation(model, name) do
  Axon.layer(
    fn input, _opts, state ->
      # Store activation in state under :captures key
      captures = Map.get(state, :captures, %{})
      updated_captures = Map.put(captures, name, input)
      updated_state = Map.put(state, :captures, updated_captures)

      # Return input unchanged (identity operation)
      {input, updated_state}
    end,
    [model],
    name: :"capture_#{name}"
  )
end

@doc """
Extract captured activations from model state after forward pass.
"""
@spec extract_captures(map()) :: map()
def extract_captures(model_state) do
  Map.get(model_state, :captures, %{})
end
```

### Activity Regularization Train Step

```elixir
@doc """
Build training step with activity regularization on captured layers.

## Parameters

  * `model` - Axon model with capture layers
  * `base_loss_fn` - Task loss function
  * `activity_penalties` - Map of capture name to penalty config
  * `optimizer` - Polaris optimizer

## Example

    activity_penalties = %{
      layer1: {&NxPenalties.Penalties.l1/2, weight: 0.001},
      layer2: {&NxPenalties.Penalties.l1/2, weight: 0.0005}
    }

    train_step = NxPenalties.Integration.Axon.build_activity_train_step(
      model,
      &Axon.Losses.categorical_cross_entropy/2,
      activity_penalties,
      Polaris.Optimizers.adam()
    )
"""
@spec build_activity_train_step(Axon.t(), function(), map(), term()) :: function()
def build_activity_train_step(model, base_loss_fn, activity_penalties, optimizer) do
  {init_fn, predict_fn} = Axon.build(model, mode: :train)
  {opt_init_fn, opt_update_fn} = optimizer

  init = fn data, init_state ->
    params = init_fn.(data, init_state)
    opt_state = opt_init_fn.(params)
    %{model_state: params, optimizer_state: opt_state}
  end

  step = fn state, batch ->
    %{model_state: params, optimizer_state: opt_state} = state
    {x, y_true} = batch

    # Forward pass with state to capture activations
    {loss, grads} = Nx.Defn.value_and_grad(params, fn p ->
      {y_pred, model_state} = predict_fn.(p, x)

      # Base loss
      base_loss = base_loss_fn.(y_true, y_pred)

      # Activity penalties from captured layers
      captures = extract_captures(model_state)
      activity_loss =
        activity_penalties
        |> Enum.map(fn {name, {penalty_fn, opts}} ->
          case Map.get(captures, name) do
            nil -> Nx.tensor(0.0)
            activation ->
              weight = Keyword.get(opts, :weight, 1.0)
              Nx.multiply(penalty_fn.(activation, opts), weight)
          end
        end)
        |> Enum.reduce(Nx.tensor(0.0), &Nx.add/2)

      Nx.add(base_loss, activity_loss)
    end)

    # Update params
    {updates, new_opt_state} = opt_update_fn.(grads, opt_state, params)
    new_params = Polaris.Updates.apply_updates(params, updates)

    {%{model_state: new_params, optimizer_state: new_opt_state}, %{"loss" => loss}}
  end

  {init, step}
end
```

---

## Loop Handlers

### Metric Logging

```elixir
@doc """
Add penalty metrics logging to an Axon loop.

## Example

    loop =
      Axon.Loop.trainer(model, loss, optimizer)
      |> NxPenalties.Integration.Axon.log_penalties(pipeline)
"""
@spec log_penalties(Axon.Loop.t(), NxPenalties.Pipeline.t()) :: Axon.Loop.t()
def log_penalties(loop, pipeline) do
  penalty_names = Enum.map(pipeline.entries, fn {name, _, _, _, _} -> name end)

  Enum.reduce(penalty_names, loop, fn name, acc ->
    acc
    |> Axon.Loop.metric(String.to_atom("#{name}"), "#{name}_loss")
    |> Axon.Loop.metric(String.to_atom("#{name}_weighted"), "#{name}_weighted")
  end)
end
```

### Callback for Dynamic Weights

```elixir
@doc """
Add a callback to update pipeline weights during training.

Useful for curriculum learning or scheduled regularization.

## Example

    schedule_fn = fn epoch ->
      kl_weight = min(epoch / 10, 1.0) * 0.1
      %{kl: kl_weight}
    end

    loop =
      Axon.Loop.trainer(model, loss, optimizer)
      |> NxPenalties.Integration.Axon.schedule_weights(pipeline, schedule_fn)
"""
@spec schedule_weights(Axon.Loop.t(), NxPenalties.Pipeline.t(), function()) :: Axon.Loop.t()
def schedule_weights(loop, pipeline, schedule_fn) do
  Axon.Loop.handle(loop, :epoch_started, fn state ->
    epoch = state.epoch
    weight_updates = schedule_fn.(epoch)

    updated_pipeline =
      Enum.reduce(weight_updates, pipeline, fn {name, weight}, acc ->
        NxPenalties.Pipeline.update_weight(acc, name, weight)
      end)

    # Store updated pipeline in state
    {:continue, put_in(state[:handler_metadata][:pipeline], updated_pipeline)}
  end)
end
```

---

## Usage Examples

### Simple L2 Regularization

```elixir
defmodule SimpleRegularization do
  def train(model, data) do
    loss = NxPenalties.Integration.Axon.wrap_loss(
      &Axon.Losses.mean_squared_error/2,
      &NxPenalties.Penalties.l2/2,
      lambda: 0.01
    )

    model
    |> Axon.Loop.trainer(loss, Polaris.Optimizers.adam())
    |> Axon.Loop.run(data, %{}, epochs: 10)
  end
end
```

### Multi-Penalty Training

```elixir
defmodule MultiPenaltyTraining do
  def train(model, data) do
    pipeline =
      NxPenalties.Pipeline.new()
      |> NxPenalties.Pipeline.add(:l2, &NxPenalties.Penalties.l2/2, weight: 0.01)
      |> NxPenalties.Pipeline.add(:entropy, &NxPenalties.Divergences.entropy/2,
           weight: 0.001, opts: [mode: :penalty])

    {init, step} = NxPenalties.Integration.Axon.build_train_step(
      model,
      &Axon.Losses.categorical_cross_entropy/2,
      pipeline,
      Polaris.Optimizers.adam()
    )

    Axon.Loop.loop(step, init)
    |> Axon.Loop.metric(:loss, "total_loss")
    |> Axon.Loop.metric(:base_loss, "base_loss")
    |> Axon.Loop.run(data, %{}, epochs: 10)
  end
end
```

### Activity Regularization

```elixir
defmodule ActivityRegularization do
  def build_model do
    Axon.input("input", shape: {nil, 784})
    |> Axon.dense(256)
    |> Axon.relu()
    |> NxPenalties.Integration.Axon.capture_activation(:hidden1)
    |> Axon.dense(128)
    |> Axon.relu()
    |> NxPenalties.Integration.Axon.capture_activation(:hidden2)
    |> Axon.dense(10)
    |> Axon.softmax()
  end

  def train(data) do
    model = build_model()

    activity_penalties = %{
      hidden1: {&NxPenalties.Penalties.l1/2, weight: 0.001},
      hidden2: {&NxPenalties.Penalties.l1/2, weight: 0.0005}
    }

    {init, step} = NxPenalties.Integration.Axon.build_activity_train_step(
      model,
      &Axon.Losses.categorical_cross_entropy/2,
      activity_penalties,
      Polaris.Optimizers.adam()
    )

    Axon.Loop.loop(step, init)
    |> Axon.Loop.run(data, %{}, epochs: 10)
  end
end
```

---

## Test Cases

```elixir
describe "wrap_loss/3" do
  test "adds penalty to loss" do
    base_loss = fn _y_true, y_pred -> Nx.mean(y_pred) end
    wrapped = NxPenalties.Integration.Axon.wrap_loss(
      base_loss,
      &NxPenalties.Penalties.l2/2,
      lambda: 0.1
    )

    y_true = Nx.tensor([1.0])
    y_pred = Nx.tensor([2.0])

    base = base_loss.(y_true, y_pred)
    total = wrapped.(y_true, y_pred)

    assert Nx.to_number(total) > Nx.to_number(base)
  end
end

describe "capture_activation/2" do
  test "captures intermediate values" do
    model =
      Axon.input("x", shape: {nil, 4})
      |> Axon.dense(8)
      |> NxPenalties.Integration.Axon.capture_activation(:hidden)
      |> Axon.dense(2)

    {init_fn, predict_fn} = Axon.build(model, mode: :train)
    params = init_fn.(Nx.template({1, 4}, :f32), %{})

    input = Nx.random_uniform({1, 4})
    {_output, state} = predict_fn.(params, input)

    captures = NxPenalties.Integration.Axon.extract_captures(state)
    assert Map.has_key?(captures, :hidden)
    assert Nx.shape(captures.hidden) == {1, 8}
  end
end
```

---

## Integration Checklist

- [ ] wrap_loss/3 works with standard Axon training
- [ ] Pipeline integration provides metrics access
- [ ] Activity regularization captures work correctly
- [ ] State threading handles functional constraints
- [ ] Loop handlers integrate with Axon.Loop
- [ ] Curriculum learning pattern supported
- [ ] Documentation covers all three patterns
- [ ] Examples are runnable and tested
