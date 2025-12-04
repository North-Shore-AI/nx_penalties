# 08: API Reference Specification

## Overview

This document provides the complete API specification for NxPenalties. All public functions, their signatures, options, and behaviors are documented here.

## Module: NxPenalties

The main entry module with validated public API.

```elixir
defmodule NxPenalties do
  @moduledoc """
  Composable regularization penalties for the Nx ecosystem.

  NxPenalties provides pure Nx.Defn functions for regularization,
  plus composition infrastructure and framework integrations.

  ## Quick Start

      # Simple penalty
      loss = NxPenalties.l1(params, lambda: 0.01)

      # Pipeline composition
      pipeline =
        NxPenalties.pipeline([
          {:l1, weight: 0.001},
          {:l2, weight: 0.01},
          {:entropy, weight: 0.1, mode: :bonus}
        ])

      {total, metrics} = NxPenalties.compute(pipeline, tensor)

  ## Modules

    * `NxPenalties.Penalties` - L1, L2, Elastic Net
    * `NxPenalties.Divergences` - KL, JS, Entropy
    * `NxPenalties.Constraints` - Orthogonality, Gradient Penalty
    * `NxPenalties.Pipeline` - Composition engine
    * `NxPenalties.Integration.Axon` - Axon training helpers
    * `NxPenalties.Integration.Polaris` - Gradient transforms
  """
end
```

---

## Penalties API

### NxPenalties.l1/2

```elixir
@doc """
L1 penalty (Lasso regularization).

## Examples

    iex> NxPenalties.l1(Nx.tensor([1.0, -2.0, 3.0]), lambda: 0.1)
    #Nx.Tensor<f32: 0.6>

## Options

  * `:lambda` - Regularization strength (default: `0.01`)
  * `:reduction` - `:sum` or `:mean` (default: `:sum`)

## Returns

Scalar tensor with penalty value.
"""
@spec l1(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
def l1(tensor, opts \\ [])
```

### NxPenalties.l2/2

```elixir
@doc """
L2 penalty (Ridge regularization).

## Examples

    iex> NxPenalties.l2(Nx.tensor([1.0, 2.0, 3.0]), lambda: 0.1)
    #Nx.Tensor<f32: 1.4>

## Options

  * `:lambda` - Regularization strength (default: `0.01`)
  * `:reduction` - `:sum` or `:mean` (default: `:sum`)
  * `:clip` - Max absolute value before squaring (default: `nil`)

## Returns

Scalar tensor with penalty value.
"""
@spec l2(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
def l2(tensor, opts \\ [])
```

### NxPenalties.elastic_net/2

```elixir
@doc """
Elastic Net penalty (combined L1 + L2).

## Examples

    iex> NxPenalties.elastic_net(Nx.tensor([1.0, 2.0]), lambda: 0.1, l1_ratio: 0.5)
    #Nx.Tensor<f32: 0.4>

## Options

  * `:lambda` - Overall strength (default: `0.01`)
  * `:l1_ratio` - L1/L2 balance, 1.0 = pure L1 (default: `0.5`)
  * `:reduction` - `:sum` or `:mean` (default: `:sum`)

## Returns

Scalar tensor with penalty value.
"""
@spec elastic_net(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
def elastic_net(tensor, opts \\ [])
```

---

## Divergences API

### NxPenalties.kl_divergence/3

```elixir
@doc """
Kullback-Leibler divergence: KL(P || Q).

Both inputs should be log-probabilities.

## Examples

    iex> p = Nx.tensor([[-1.0, -2.0, -3.0]])
    iex> q = Nx.tensor([[-1.5, -1.5, -1.5]])
    iex> NxPenalties.kl_divergence(p, q)
    #Nx.Tensor<f32: ...>

## Options

  * `:reduction` - `:mean`, `:sum`, or `:none` (default: `:mean`)

## Returns

Scalar tensor (or per-sample with `:none`).
"""
@spec kl_divergence(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
def kl_divergence(p_logprobs, q_logprobs, opts \\ [])
```

### NxPenalties.js_divergence/3

```elixir
@doc """
Jensen-Shannon divergence (symmetric KL).

## Options

  * `:reduction` - `:mean`, `:sum`, or `:none` (default: `:mean`)

## Returns

Scalar tensor. Bounded in [0, log(2)].
"""
@spec js_divergence(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
def js_divergence(p_logprobs, q_logprobs, opts \\ [])
```

### NxPenalties.entropy/2

```elixir
@doc """
Shannon entropy of a distribution.

## Examples

    iex> logprobs = Nx.tensor([[-1.1, -1.1, -1.1, -1.1]])  # ~uniform
    iex> NxPenalties.entropy(logprobs, normalize: true)
    #Nx.Tensor<f32: ~1.0>

## Options

  * `:mode` - `:penalty` (minimize entropy) or `:bonus` (maximize) (default: `:penalty`)
  * `:reduction` - `:mean`, `:sum`, or `:none` (default: `:mean`)
  * `:normalize` - Normalize by max entropy (default: `false`)

## Returns

Scalar tensor.
"""
@spec entropy(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
def entropy(logprobs, opts \\ [])
```

---

## Constraints API

### NxPenalties.orthogonality/2

```elixir
@doc """
Orthogonality penalty for uncorrelated representations.

## Options

  * `:mode` - `:soft` (off-diagonal only) or `:hard` (default: `:soft`)
  * `:normalize` - Normalize rows before Gram matrix (default: `true`)

## Returns

Scalar penalty value.
"""
@spec orthogonality(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
def orthogonality(tensor, opts \\ [])
```

### NxPenalties.consistency/3

```elixir
@doc """
Consistency penalty between paired outputs.

## Options

  * `:metric` - `:mse`, `:l1`, `:kl`, or `:cosine` (default: `:mse`)
  * `:reduction` - `:mean`, `:sum`, or `:none` (default: `:mean`)

## Returns

Scalar divergence value.
"""
@spec consistency(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
def consistency(output1, output2, opts \\ [])
```

---

## Pipeline API

### NxPenalties.Pipeline.new/1

```elixir
@doc """
Create a new empty pipeline.

## Options

  * `:reduction` - `:sum` or `:mean` (default: `:sum`)
  * `:scale` - Global scale factor (default: `1.0`)
"""
@spec new(keyword()) :: Pipeline.t()
def new(opts \\ [])
```

### NxPenalties.Pipeline.add/4

```elixir
@doc """
Add a penalty to the pipeline.

## Parameters

  * `pipeline` - Pipeline struct
  * `name` - Atom identifier
  * `penalty_fn` - Function `(tensor, opts) -> scalar`
  * `opts` - Configuration:
    * `:weight` - Multiplier (default: `1.0`)
    * `:opts` - Penalty function options
    * `:enabled` - Active flag (default: `true`)
"""
@spec add(Pipeline.t(), atom(), function(), keyword()) :: Pipeline.t()
def add(pipeline, name, penalty_fn, opts \\ [])
```

### NxPenalties.Pipeline.compute/3

```elixir
@doc """
Execute pipeline and return total + metrics.

## Returns

`{total_tensor, metrics_map}` where metrics contains:
- `"{name}"` - Raw penalty value
- `"{name}_weighted"` - Weighted value
- `"total"` - Sum of weighted penalties
"""
@spec compute(Pipeline.t(), Nx.Tensor.t(), keyword()) :: {Nx.Tensor.t(), map()}
def compute(pipeline, tensor, opts \\ [])
```

---

## Integration API

### NxPenalties.Integration.Axon

```elixir
# Loss wrapping
@spec wrap_loss(function(), function(), keyword()) :: function()
def wrap_loss(base_loss_fn, penalty_fn, opts \\ [])

@spec wrap_loss_with_pipeline(function(), Pipeline.t(), keyword()) :: function()
def wrap_loss_with_pipeline(base_loss_fn, pipeline, opts \\ [])

# Train step building
@spec build_train_step(Axon.t(), function(), Pipeline.t(), term()) :: {function(), function()}
def build_train_step(model, base_loss_fn, pipeline, optimizer)

# Activity regularization
@spec capture_activation(Axon.t(), atom()) :: Axon.t()
def capture_activation(model, name)

@spec extract_captures(map()) :: map()
def extract_captures(model_state)
```

### NxPenalties.Integration.Polaris

```elixir
# Gradient transforms
@spec add_l1_decay(term(), float()) :: {function(), function()}
def add_l1_decay(optimizer, decay \\ 0.001)

@spec add_l2_decay(term(), float()) :: {function(), function()}
def add_l2_decay(optimizer, decay \\ 0.01)

@spec add_elastic_decay(term(), float(), keyword()) :: {function(), function()}
def add_elastic_decay(optimizer, decay \\ 0.01, opts \\ [])

@spec add_gradient_clipping(term(), float()) :: {function(), function()}
def add_gradient_clipping(optimizer, max_norm \\ 1.0)
```

---

## Type Specifications

```elixir
# Core types
@type tensor :: Nx.Tensor.t()
@type opts :: keyword()

# Pipeline types
@type penalty_fn :: (tensor(), opts() -> tensor())
@type entry :: {atom(), penalty_fn(), number(), opts()}

@type pipeline :: %Pipeline{
  entries: [entry()],
  reduction: :sum | :mean,
  scale: number() | tensor()
}

# Integration types
@type loss_fn :: (tensor(), tensor() -> tensor())
@type optimizer :: {function(), function()}
@type train_step :: {function(), function()}
```

---

## Error Handling

### Validation Errors

```elixir
# Raised when options are invalid
defexception NxPenalties.ValidationError, [:message, :key, :value]

# Example
raise NxPenalties.ValidationError,
  message: "Invalid reduction option",
  key: :reduction,
  value: :invalid
```

### Numerical Errors

Functions do not raise on numerical issues (NaN/Inf). Instead:

1. **NaN propagation** - NaN inputs produce NaN outputs
2. **Inf handling** - Inf values are preserved or clipped based on function
3. **Validation helpers** - Use `NxPenalties.validate/1` to check tensors

```elixir
@doc """
Validate tensor is finite (no NaN or Inf).

Returns `{:ok, tensor}` or `{:error, :nan | :inf}`.
"""
@spec validate(tensor()) :: {:ok, tensor()} | {:error, atom()}
def validate(tensor)
```

---

## Telemetry Events

| Event | Measurements | Metadata |
|-------|-------------|----------|
| `[:nx_penalties, :penalty, :compute, :start]` | `%{system_time: t}` | `%{name: atom}` |
| `[:nx_penalties, :penalty, :compute, :stop]` | `%{duration: ns, value: float}` | `%{name: atom}` |
| `[:nx_penalties, :pipeline, :compute, :start]` | `%{system_time: t}` | `%{size: int}` |
| `[:nx_penalties, :pipeline, :compute, :stop]` | `%{duration: ns}` | `%{metrics: map}` |

---

## Version Compatibility

| Dependency | Version | Notes |
|------------|---------|-------|
| Elixir | >= 1.14 | Required for Nx |
| Nx | ~> 0.9 | Core dependency |
| Axon | ~> 0.6 | Optional, for integration |
| Polaris | ~> 0.1 | Optional, for integration |
| EXLA | >= 0.0.0 | Optional, test only |
