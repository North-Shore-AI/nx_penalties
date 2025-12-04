# 06: Polaris Integration Implementation Specification

## Overview

Polaris is Axon's optimization library, extracted to allow independent use. It provides composable optimizer transformations similar to Optax in JAX. This document specifies gradient-space regularization transforms that complement the loss-space penalties in the core library.

## Gradient vs Loss Regularization

| Approach | Where Applied | Example | NxPenalties Module |
|----------|--------------|---------|-------------------|
| **Loss-space** | Added to loss function | L1 on activations | `NxPenalties.Penalties` |
| **Gradient-space** | Transform gradients | Weight decay | `NxPenalties.Integration.Polaris` |

Gradient-space regularization modifies the parameter updates directly rather than contributing to the loss. This is the standard approach for weight decay in modern optimizers like AdamW.

## Module: NxPenalties.Integration.Polaris

### File Location
```
lib/nx_penalties/integration/polaris.ex
```

### Module Structure
```elixir
defmodule NxPenalties.Integration.Polaris do
  @moduledoc """
  Gradient transformations for use with Polaris optimizers.

  These transforms operate on gradients and parameters, not on
  the loss function. They follow Polaris's composable pattern.

  ## Composition

  Polaris transforms compose via piping:

      optimizer =
        Polaris.Optimizers.adam(learning_rate: 0.001)
        |> NxPenalties.Integration.Polaris.add_l1_decay(0.0001)
        |> NxPenalties.Integration.Polaris.add_gradient_clipping(1.0)

  ## Weight Decay vs L2 Regularization

  These are mathematically equivalent for SGD but differ for
  adaptive optimizers like Adam. Weight decay (implemented here)
  is generally preferred for modern training.

  Reference: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017)
  """

  import Nx.Defn
end
```

---

## Function: add_l2_decay/2

```elixir
@doc """
Add L2 weight decay to gradients.

This is equivalent to Polaris's built-in weight decay in AdamW,
provided for completeness and explicit composition.

Weight decay modifies the gradient: g' = g + λw
Where λ is the decay rate and w is the weight.

## Parameters

  * `optimizer_or_transform` - Base optimizer or transform
  * `decay` - Decay rate. Default: `0.01`

## Note

For AdamW, prefer using the built-in `:decay` option:

    Polaris.Optimizers.adamw(learning_rate: 0.001, decay: 0.01)

This transform is useful when you want to add decay to an optimizer
that doesn't have it built-in, or for explicit composition.

## Example

    optimizer =
      Polaris.Optimizers.adam(learning_rate: 0.001)
      |> NxPenalties.Integration.Polaris.add_l2_decay(0.01)
"""
@spec add_l2_decay(term(), float()) :: {function(), function()}
def add_l2_decay(optimizer_or_transform, decay \\ 0.01) do
  {base_init, base_update} = normalize_to_transform(optimizer_or_transform)

  init_fn = fn params ->
    base_state = base_init.(params)
    %{base: base_state, l2_decay: decay}
  end

  update_fn = fn gradients, state, params ->
    # Add decay to gradients: g' = g + λw
    decayed_grads = deep_map(gradients, params, fn g, w ->
      Nx.add(g, Nx.multiply(w, state.l2_decay))
    end)

    # Apply base optimizer
    {updates, new_base_state} = base_update.(decayed_grads, state.base, params)

    new_state = %{state | base: new_base_state}
    {updates, new_state}
  end

  {init_fn, update_fn}
end
```

---

## Function: add_l1_decay/2

```elixir
@doc """
Add L1 weight decay (sign decay) to gradients.

Modifies the gradient: g' = g + λ * sign(w)
This encourages sparsity in the weights.

## Parameters

  * `optimizer_or_transform` - Base optimizer or transform
  * `decay` - Decay rate. Default: `0.001`

## Note

L1 decay can cause weights to oscillate around zero. Consider
using a small threshold to zero out very small weights.

## Example

    optimizer =
      Polaris.Optimizers.sgd(learning_rate: 0.01)
      |> NxPenalties.Integration.Polaris.add_l1_decay(0.001)
"""
@spec add_l1_decay(term(), float()) :: {function(), function()}
def add_l1_decay(optimizer_or_transform, decay \\ 0.001) do
  {base_init, base_update} = normalize_to_transform(optimizer_or_transform)

  init_fn = fn params ->
    base_state = base_init.(params)
    %{base: base_state, l1_decay: decay}
  end

  update_fn = fn gradients, state, params ->
    # Add L1 decay: g' = g + λ * sign(w)
    decayed_grads = deep_map(gradients, params, fn g, w ->
      Nx.add(g, Nx.multiply(Nx.sign(w), state.l1_decay))
    end)

    {updates, new_base_state} = base_update.(decayed_grads, state.base, params)

    new_state = %{state | base: new_base_state}
    {updates, new_state}
  end

  {init_fn, update_fn}
end
```

---

## Function: add_elastic_decay/3

```elixir
@doc """
Add elastic net (L1 + L2) weight decay to gradients.

Combines L1 and L2 decay:
g' = g + λ * (α * sign(w) + (1-α) * w)

## Parameters

  * `optimizer_or_transform` - Base optimizer or transform
  * `decay` - Overall decay rate. Default: `0.01`
  * `l1_ratio` - Ratio of L1 to L2 (α). Default: `0.5`

## Example

    optimizer =
      Polaris.Optimizers.adam(learning_rate: 0.001)
      |> NxPenalties.Integration.Polaris.add_elastic_decay(0.01, l1_ratio: 0.3)
"""
@spec add_elastic_decay(term(), float(), keyword()) :: {function(), function()}
def add_elastic_decay(optimizer_or_transform, decay \\ 0.01, opts \\ []) do
  l1_ratio = Keyword.get(opts, :l1_ratio, 0.5)

  {base_init, base_update} = normalize_to_transform(optimizer_or_transform)

  init_fn = fn params ->
    base_state = base_init.(params)
    %{base: base_state, decay: decay, l1_ratio: l1_ratio}
  end

  update_fn = fn gradients, state, params ->
    alpha = state.l1_ratio
    lambda = state.decay

    decayed_grads = deep_map(gradients, params, fn g, w ->
      l1_term = Nx.multiply(Nx.sign(w), alpha)
      l2_term = Nx.multiply(w, 1.0 - alpha)
      decay_term = Nx.multiply(Nx.add(l1_term, l2_term), lambda)
      Nx.add(g, decay_term)
    end)

    {updates, new_base_state} = base_update.(decayed_grads, state.base, params)

    new_state = %{state | base: new_base_state}
    {updates, new_state}
  end

  {init_fn, update_fn}
end
```

---

## Function: add_gradient_clipping/2

```elixir
@doc """
Add gradient norm clipping.

Clips gradients to have a maximum L2 norm. This is a form of
regularization that prevents exploding gradients.

## Parameters

  * `optimizer_or_transform` - Base optimizer or transform
  * `max_norm` - Maximum gradient norm. Default: `1.0`

## Example

    optimizer =
      Polaris.Optimizers.adam(learning_rate: 0.001)
      |> NxPenalties.Integration.Polaris.add_gradient_clipping(1.0)
"""
@spec add_gradient_clipping(term(), float()) :: {function(), function()}
def add_gradient_clipping(optimizer_or_transform, max_norm \\ 1.0) do
  {base_init, base_update} = normalize_to_transform(optimizer_or_transform)

  init_fn = fn params ->
    base_state = base_init.(params)
    %{base: base_state, max_norm: max_norm}
  end

  update_fn = fn gradients, state, params ->
    # Compute global gradient norm
    grad_norm = compute_global_norm(gradients)

    # Clip if necessary
    clip_factor = Nx.min(Nx.divide(state.max_norm, Nx.max(grad_norm, 1.0e-8)), 1.0)
    clipped_grads = deep_map_single(gradients, fn g ->
      Nx.multiply(g, clip_factor)
    end)

    {updates, new_base_state} = base_update.(clipped_grads, state.base, params)

    new_state = %{state | base: new_base_state}
    {updates, new_state}
  end

  {init_fn, update_fn}
end

defnp compute_global_norm(gradients) do
  # Sum of squared norms across all parameters
  gradients
  |> flatten_params()
  |> Enum.map(&Nx.sum(Nx.power(&1, 2)))
  |> Enum.reduce(&Nx.add/2)
  |> Nx.sqrt()
end
```

---

## Function: add_gradient_noise/3

```elixir
@doc """
Add Gaussian noise to gradients.

A form of regularization that can help escape local minima.
Implements the schedule from "Adding Gradient Noise Improves
Learning for Very Deep Networks" (Neelakantan et al., 2015).

## Parameters

  * `optimizer_or_transform` - Base optimizer or transform
  * `variance` - Base noise variance. Default: `0.01`
  * `opts` - Options:
    * `:decay` - Variance decay rate. Default: `0.55`
    * `:key` - PRNG key. Default: auto-generated

## Example

    optimizer =
      Polaris.Optimizers.sgd(learning_rate: 0.01)
      |> NxPenalties.Integration.Polaris.add_gradient_noise(0.01)
"""
@spec add_gradient_noise(term(), float(), keyword()) :: {function(), function()}
def add_gradient_noise(optimizer_or_transform, variance \\ 0.01, opts \\ []) do
  decay = Keyword.get(opts, :decay, 0.55)

  {base_init, base_update} = normalize_to_transform(optimizer_or_transform)

  init_fn = fn params ->
    base_state = base_init.(params)
    %{
      base: base_state,
      variance: variance,
      decay: decay,
      step: 0,
      key: Nx.Random.key(System.unique_integer())
    }
  end

  update_fn = fn gradients, state, params ->
    # Compute current variance: σ² = η / (1 + t)^γ
    current_var = state.variance / :math.pow(1 + state.step, state.decay)

    # Add noise to each gradient
    {noisy_grads, new_key} = add_noise_to_grads(gradients, current_var, state.key)

    {updates, new_base_state} = base_update.(noisy_grads, state.base, params)

    new_state = %{state |
      base: new_base_state,
      step: state.step + 1,
      key: new_key
    }

    {updates, new_state}
  end

  {init_fn, update_fn}
end

defnp add_noise_to_grads(gradients, variance, key) do
  # Implementation using Nx.Random
  # Returns {noisy_gradients, new_key}
end
```

---

## Helper Functions

```elixir
# Normalize optimizer or transform to {init, update} tuple
defp normalize_to_transform({init, update}) when is_function(init) and is_function(update) do
  {init, update}
end

defp normalize_to_transform(optimizer) when is_tuple(optimizer) do
  optimizer
end

# Deep map over two nested structures (gradients and params)
defp deep_map(gradients, params, fun) when is_map(gradients) and is_map(params) do
  Map.new(gradients, fn {key, g} ->
    p = Map.fetch!(params, key)
    {key, deep_map(g, p, fun)}
  end)
end

defp deep_map(gradient, param, fun) do
  fun.(gradient, param)
end

# Deep map over single nested structure
defp deep_map_single(gradients, fun) when is_map(gradients) do
  Map.new(gradients, fn {key, g} ->
    {key, deep_map_single(g, fun)}
  end)
end

defp deep_map_single(gradient, fun) do
  fun.(gradient)
end

# Flatten nested params to list
defp flatten_params(params) when is_map(params) do
  Enum.flat_map(params, fn {_key, v} -> flatten_params(v) end)
end

defp flatten_params(tensor) do
  [Nx.flatten(tensor)]
end
```

---

## Usage Examples

### Basic Weight Decay

```elixir
# Using built-in AdamW
optimizer = Polaris.Optimizers.adamw(learning_rate: 0.001, decay: 0.01)

# Or explicit composition
optimizer =
  Polaris.Optimizers.adam(learning_rate: 0.001)
  |> NxPenalties.Integration.Polaris.add_l2_decay(0.01)
```

### L1 + Gradient Clipping

```elixir
optimizer =
  Polaris.Optimizers.adam(learning_rate: 0.001)
  |> NxPenalties.Integration.Polaris.add_l1_decay(0.0001)
  |> NxPenalties.Integration.Polaris.add_gradient_clipping(1.0)
```

### Full Regularization Stack

```elixir
optimizer =
  Polaris.Optimizers.adam(learning_rate: 0.001)
  |> NxPenalties.Integration.Polaris.add_elastic_decay(0.01, l1_ratio: 0.1)
  |> NxPenalties.Integration.Polaris.add_gradient_clipping(5.0)
  |> NxPenalties.Integration.Polaris.add_gradient_noise(0.001)
```

---

## Test Cases

```elixir
describe "add_l2_decay/2" do
  test "modifies gradients correctly" do
    optimizer =
      Polaris.Optimizers.sgd(learning_rate: 0.1)
      |> NxPenalties.Integration.Polaris.add_l2_decay(0.1)

    {init_fn, update_fn} = optimizer

    params = %{w: Nx.tensor([1.0, 2.0])}
    gradients = %{w: Nx.tensor([0.1, 0.1])}

    state = init_fn.(params)
    {updates, _new_state} = update_fn.(gradients, state, params)

    # Gradient should be: 0.1 + 0.1 * [1, 2] = [0.2, 0.3]
    # Update with lr=0.1: -0.1 * [0.2, 0.3] = [-0.02, -0.03]
    assert_close(updates.w, Nx.tensor([-0.02, -0.03]))
  end
end

describe "add_l1_decay/2" do
  test "uses sign of weights" do
    optimizer =
      Polaris.Optimizers.sgd(learning_rate: 0.1)
      |> NxPenalties.Integration.Polaris.add_l1_decay(0.1)

    {init_fn, update_fn} = optimizer

    params = %{w: Nx.tensor([2.0, -3.0])}
    gradients = %{w: Nx.tensor([0.0, 0.0])}

    state = init_fn.(params)
    {updates, _new_state} = update_fn.(gradients, state, params)

    # Gradient: 0 + 0.1 * sign([2, -3]) = [0.1, -0.1]
    # Update: -0.1 * [0.1, -0.1] = [-0.01, 0.01]
    assert_close(updates.w, Nx.tensor([-0.01, 0.01]))
  end
end

describe "add_gradient_clipping/2" do
  test "clips large gradients" do
    optimizer =
      Polaris.Optimizers.sgd(learning_rate: 1.0)
      |> NxPenalties.Integration.Polaris.add_gradient_clipping(1.0)

    {init_fn, update_fn} = optimizer

    params = %{w: Nx.tensor([0.0, 0.0])}
    gradients = %{w: Nx.tensor([3.0, 4.0])}  # norm = 5

    state = init_fn.(params)
    {updates, _new_state} = update_fn.(gradients, state, params)

    # Clipped to norm 1: [0.6, 0.8]
    # Update: -1.0 * [0.6, 0.8]
    clipped_norm = Nx.sqrt(Nx.sum(Nx.power(updates.w, 2)))
    assert_close(clipped_norm, Nx.tensor(1.0), atol: 1.0e-5)
  end

  test "doesn't clip small gradients" do
    optimizer =
      Polaris.Optimizers.sgd(learning_rate: 1.0)
      |> NxPenalties.Integration.Polaris.add_gradient_clipping(10.0)

    {init_fn, update_fn} = optimizer

    params = %{w: Nx.tensor([0.0, 0.0])}
    gradients = %{w: Nx.tensor([1.0, 1.0])}  # norm ≈ 1.41 < 10

    state = init_fn.(params)
    {updates, _new_state} = update_fn.(gradients, state, params)

    assert_close(updates.w, Nx.tensor([-1.0, -1.0]))
  end
end
```

---

## Integration Checklist

- [ ] All transforms follow Polaris {init, update} pattern
- [ ] Transforms compose correctly with piping
- [ ] Deep map handles nested parameter structures
- [ ] Gradient clipping computes global norm correctly
- [ ] L1 decay handles zero weights
- [ ] Noise injection is stateful (key management)
- [ ] Documentation explains gradient vs loss regularization
- [ ] Examples show composition patterns
