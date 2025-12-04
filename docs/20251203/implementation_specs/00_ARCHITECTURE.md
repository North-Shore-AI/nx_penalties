# 00: Core Architecture

## Design Philosophy

NxPenalties follows three core principles:

1. **Pure Functions in Defn** - All numerical computation happens inside `Nx.Defn` for JIT compilation
2. **Composability Over Configuration** - Small functions combine to form complex objectives
3. **Separation of Concerns** - Math is separate from integration glue

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           User Code / Training Loop                      │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        NxPenalties.Pipeline                              │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  Composition Engine: Weights, Reduction, Multi-Objective            ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ NxPenalties      │ │ NxPenalties      │ │ NxPenalties      │
│ .Penalties       │ │ .Divergences     │ │ .Constraints     │
│                  │ │                  │ │                  │
│ • l1/2           │ │ • kl_divergence/3│ │ • orthogonality/2│
│ • l2/2           │ │ • js_divergence/3│ │ • consistency/3  │
│ • elastic_net/2  │ │ • entropy/2      │ │ • orthogonality/2│
│                  │ │                  │ │ (GradientPenalty)│
└──────────────────┘ └──────────────────┘ └──────────────────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              Nx.Defn                                     │
│                    (JIT Compilation Boundary)                            │
└─────────────────────────────────────────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
    ┌──────────┐         ┌──────────┐         ┌──────────┐
    │   EXLA   │         │  Torchx  │         │  Binary  │
    │  (XLA)   │         │(PyTorch) │         │ Backend  │
    └──────────┘         └──────────┘         └──────────┘
```

## Module Responsibilities

### NxPenalties.Penalties

**Purpose**: Parameter-based penalties (operate on weights or outputs)

**Functions**:
- `l1/2` - L1 norm (sparsity inducing)
- `l2/2` - L2 norm (smoothing)
- `elastic_net/2` - Weighted combination

**Signature Pattern**:
```elixir
@spec l1(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
defn l1(tensor, opts \\ []) do
  lambda = opts[:lambda] || 0.01
  tensor |> Nx.abs() |> Nx.sum() |> Nx.multiply(lambda)
end
```

### NxPenalties.Divergences

**Purpose**: Distribution-based penalties (operate on probability distributions)

**Functions**:
- `kl_divergence/3` - KL(P||Q)
- `js_divergence/3` - Jensen-Shannon (symmetric KL)
- `entropy/2` - Shannon entropy (maximize or minimize)

**Signature Pattern**:
```elixir
@spec kl_divergence(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
defn kl_divergence(p_logprobs, q_logprobs, opts \\ []) do
  # Returns scalar KL divergence
end
```

### NxPenalties.Constraints

**Purpose**: Structural constraints on representations

**Functions**:
- `orthogonality/2` - Penalize correlated dimensions
- `consistency/3` - Penalize divergence between paired inputs

**Complexity**: These require more sophisticated tensor operations and may need gradients of gradients.

### NxPenalties.GradientPenalty

**Purpose**: Lipschitz-style gradient control (tensor-only)

**Functions**:
- `gradient_penalty/3` - Gradient norm penalty (expensive)
- `interpolated_gradient_penalty/4` - WGAN-GP style interpolation
- `output_magnitude_penalty/2` - Cheaper proxy without second-order grads

### NxPenalties.Pipeline

**Purpose**: Compose multiple penalties into single objective

**Core Struct**:
```elixir
defmodule NxPenalties.Pipeline do
  defstruct [:penalties, :weights, :reduction]

  @type t :: %__MODULE__{
    penalties: [{atom(), function(), keyword()}],
    weights: [float()],
    reduction: :sum | :mean | :weighted_mean
  }
end
```

**Builder Pattern**:
```elixir
pipeline =
  NxPenalties.Pipeline.new()
  |> NxPenalties.Pipeline.add(:l1, &NxPenalties.Penalties.l1/2, weight: 0.001)
  |> NxPenalties.Pipeline.add(:l2, &NxPenalties.Penalties.l2/2, weight: 0.01)
  |> NxPenalties.Pipeline.add(:entropy, &NxPenalties.Divergences.entropy/2,
       weight: 0.1, opts: [mode: :penalty])
```

**Execution**:
```elixir
# Returns {total_loss, metrics_map}
{loss, metrics} = NxPenalties.Pipeline.compute(pipeline, tensor, opts)

# NxPenalties pipelines are single-tensor; data-aware adapters live in Tinkex.
```

### NxPenalties.Integration.Axon

**Purpose**: Glue code for Axon training loops

**Functions**:
- `wrap_loss/3` - Wrap base loss with penalty additions
- `build_step/4` - Generate complete train step function
- `attach_penalty/3` - Layer wrapper for activity regularization

**Key Pattern - Loss Wrapping**:
```elixir
def wrap_loss(base_loss_fn, penalty_fn, opts \\ []) do
  lambda = opts[:lambda] || 0.01

  fn y_true, y_pred ->
    base = base_loss_fn.(y_true, y_pred)
    penalty = penalty_fn.(y_pred, opts)
    Nx.add(base, Nx.multiply(penalty, lambda))
  end
end
```

**Key Pattern - Activity Regularization**:
```elixir
# Inserts identity layer that captures activation for penalty
def attach_penalty(model, penalty_fn, opts \\ []) do
  Axon.layer(model, fn input, _opts, state ->
    penalty = penalty_fn.(input, opts)
    updated_state = Map.update(state, :penalties, [penalty], &[penalty | &1])
    {input, updated_state}  # Pass through unchanged
  end)
end
```

### NxPenalties.Integration.Polaris

**Purpose**: Gradient-space transforms compatible with Polaris optimizer composition

**Functions**:
- `add_l1_decay/1` - L1 weight decay as gradient transform
- `add_l2_decay/1` - L2 weight decay (alias for existing Polaris)

**Pattern (mirrors Optax)**:
```elixir
def add_l1_decay(decay \\ 0.01) do
  {
    fn _params -> %{} end,  # init_fn (stateless)
    fn gradients, _state, params ->
      updates = deep_merge(gradients, params, fn g, p ->
        Nx.add(g, Nx.multiply(Nx.sign(p), decay))
      end)
      {updates, %{}}
    end
  }
end
```

### Telemetry

**Purpose**: Instrumentation for monitoring penalty values during training (tensor-only in NxPenalties; data-aware telemetry in Tinkex).

**NxPenalties Events**:
```elixir
[:nx_penalties, :penalty, :compute, :start]
[:nx_penalties, :penalty, :compute, :stop]
[:nx_penalties, :pipeline, :compute, :stop]
```

**Tinkex (data-aware) can emit**:
```elixir
[:tinkex, :regularizer, :gradients]
[:tinkex, :regularizer, :compute, :start|:stop|:exception]
```

## Critical Design Decisions

### Decision 1: Stateless Functions

**Choice**: All penalty functions are stateless; they take tensors and return tensors.

**Rationale**:
- Enables JIT compilation
- Avoids "functional state problem" described in research
- Composable without hidden dependencies

**Trade-off**: Cannot easily implement penalties that depend on training history (e.g., EMA of past values). Would require explicit state threading.

### Decision 2: Return Scalar by Default

**Choice**: All penalty functions return scalar tensors (single value).

**Rationale**:
- Directly usable as loss term
- Simple gradient computation
- Consistent API

**Trade-off**: If user needs per-sample penalties, they must compute manually before reduction.

### Decision 3: Options via Keyword Lists

**Choice**: Configuration through keyword opts, not struct options.

**Rationale**:
- Nx.Defn works well with keyword-based configuration
- Matches existing Nx/Axon conventions
- Simple destructuring with defaults

**Trade-off**: No compile-time validation of options (handled by nimble_options at boundary).

### Decision 4: Separate Integration Modules

**Choice**: Axon and Polaris integration in separate modules, not core.

**Rationale**:
- Core penalties work with raw Nx tensors
- Integration is optional dependency
- Can evolve independently

**Trade-off**: Users doing Axon training need to know about two modules.

## Error Handling Strategy

### Inside Defn (Hot Path)

**No exceptions**. Use:
- `Nx.select/3` for conditional behavior
- Return `Nx.Constants.nan()` or `Nx.Constants.infinity()` for invalid states
- Clipping for numerical stability

### Outside Defn (Cold Path)

**Use NimbleOptions** for validation:
```elixir
@penalty_schema [
  lambda: [type: :float, default: 0.01],
  reduction: [type: {:in, [:sum, :mean]}, default: :sum]
]

def validated_l1(tensor, opts) do
  opts = NimbleOptions.validate!(opts, @penalty_schema)
  l1(tensor, opts)
end
```

## Memory Considerations

### Large Tensor Handling

For penalties operating on large tensors (e.g., full vocabulary logprobs):

1. **Reduction early** - Don't materialize intermediate large tensors
2. **Streaming** - Use `Nx.reduce` over axes rather than full materialization
3. **Chunking** - Pipeline struct can specify chunk size for batch processing

### Gradient Memory

Gradient penalty (ADR-007) requires computing gradients of gradients:
- 2x memory for second-order gradients
- Consider checkpoint/recompute strategies
- Document memory implications prominently

## Configuration at Boundaries

```elixir
defmodule NxPenalties do
  @moduledoc """
  Entry point with validated configurations.
  """

  def l1(tensor, opts \\ []) do
    opts = validate_penalty_opts!(opts)
    NxPenalties.Penalties.l1(tensor, opts)
  end

  def pipeline(specs) when is_list(specs) do
    validated = Enum.map(specs, &validate_spec!/1)
    NxPenalties.Pipeline.new(validated)
  end
end
```

This pattern keeps defn functions pure and fast while providing user-friendly validation at the public API surface.
