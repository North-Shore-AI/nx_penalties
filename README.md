<p align="center">
  <img src="assets/nx_penalties.svg" alt="NxPenalties" width="400">
</p>

<p align="center">
  <strong>Composable Regularization Penalties for Elixir ML</strong>
</p>

<p align="center">
  <a href="https://hex.pm/packages/nx_penalties">
    <img src="https://img.shields.io/hexpm/v/nx_penalties.svg" alt="Hex.pm Version">
  </a>
  <a href="https://hexdocs.pm/nx_penalties">
    <img src="https://img.shields.io/badge/docs-hexdocs-blue.svg" alt="Documentation">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
</p>

---

## Overview

NxPenalties is a tensor-only library of regularization primitives for [Nx](https://github.com/elixir-nx/nx). It is designed to be composable inside `defn` code and training loops, leaving any data-aware adaptation (e.g., resolving references from data structures) to downstream libraries such as Tinkex.

### Features (v0.1.0)

- **Penalties**: L1, L2 (with centering/clipping), Elastic Net
- **Divergences**: KL, JS, Entropy (bonus/penalty, normalization)
- **Constraints**: Orthogonality (soft/hard/spectral, axis options), Consistency (MSE/L1/Cosine/KL)
- **Gradient Penalties**: Gradient norm, interpolated (WGAN-GP), output magnitude proxy
- **Pipeline**: Compose penalties with weights, enable/disable, gradient-compatible computation
- **Integrations**: Axon loss wrapping, Polaris gradient transforms (L1/L2/Elastic Net decay)
- **Debugging**: Gradient norm tracking, NaN/Inf validation
- **Telemetry**: Pipeline compute events

## Installation

Add `nx_penalties` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:nx_penalties, "~> 0.1.0"}
  ]
end
```

## Quick Start

### Simple Penalties

```elixir
# L1 penalty (promotes sparsity)
l1_loss = NxPenalties.l1(weights)
# => Nx.tensor(6.5)

# L2 penalty (weight decay)
l2_loss = NxPenalties.l2(weights, lambda: 0.01)
# => Nx.tensor(0.1425)

# Elastic Net (combined L1 + L2)
elastic_loss = NxPenalties.elastic_net(weights, l1_ratio: 0.5)
# => Nx.tensor(10.375)

# Add to your training loss
total_loss = Nx.add(base_loss, l1_loss)
```

### Pipeline Composition

Compose multiple penalties with individual weights:

```elixir
pipeline =
  NxPenalties.pipeline([
    {:l1, weight: 0.001},
    {:l2, weight: 0.01},
    {:entropy, weight: 0.1, opts: [mode: :bonus]}
  ])

{total_penalty, metrics} = NxPenalties.compute(pipeline, model_outputs)
total_loss = Nx.add(base_loss, total_penalty)
```

### Dynamic Weight Adjustment

Useful for curriculum learning or adaptive regularization:

```elixir
# Update weights during training
pipeline =
  pipeline
  |> NxPenalties.Pipeline.update_weight(:l1, 0.01)  # Increase L1
  |> NxPenalties.Pipeline.update_weight(:l2, 0.001) # Decrease L2

# Enable/disable penalties
pipeline = NxPenalties.Pipeline.set_enabled(pipeline, :entropy, false)
```

### Gradient-Compatible Computation

Use `compute_total/3` inside `defn`:

```elixir
total = NxPenalties.compute_total(pipeline, tensor)

grad_fn = Nx.Defn.grad(fn t -> NxPenalties.compute_total(pipeline, t) end)
gradients = grad_fn.(tensor)
```

## Divergences

For probability distributions (log-space inputs):

```elixir
# KL Divergence - knowledge distillation
kl_loss = NxPenalties.kl_divergence(student_logprobs, teacher_logprobs)

# JS Divergence - symmetric comparison
js_loss = NxPenalties.js_divergence(p_logprobs, q_logprobs)

# Entropy - encourage/discourage confidence
entropy_penalty = NxPenalties.entropy(logprobs, mode: :penalty)  # Minimize entropy
entropy_bonus = NxPenalties.entropy(logprobs, mode: :bonus)      # Maximize entropy
```

## Gradient Penalties

For Lipschitz smoothness (WGAN-GP style):

```elixir
# Full gradient penalty (expensive - use sparingly)
loss_fn = fn x -> Nx.sum(Nx.pow(x, 2)) end
gp = NxPenalties.gradient_penalty(loss_fn, tensor, target_norm: 1.0)

# Cheaper proxy - output magnitude penalty
mag_penalty = NxPenalties.output_magnitude_penalty(model_output, target: 1.0)

# Interpolated gradient penalty (WGAN-GP)
interp_gp = NxPenalties.interpolated_gradient_penalty(loss_fn, fake, real, target_norm: 1.0)
```

**Performance Warning**: Gradient penalties compute second-order derivatives and are computationally expensive. Best practices:
- Apply every N training steps instead of every step
- Use `output_magnitude_penalty/2` as a cheaper alternative
- See `examples/gradient_penalty.exs` for usage patterns

## Constraints

Structural penalties for representations:

```elixir
alias NxPenalties.Constraints

# Orthogonality - encourage uncorrelated representations
hidden_states = Nx.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])

# Soft mode: penalize off-diagonal correlations only
penalty = Constraints.orthogonality(hidden_states, mode: :soft)

# Hard mode: penalize deviation from identity matrix
penalty = Constraints.orthogonality(hidden_states, mode: :hard)

# Spectral mode: encourage uniform singular values
penalty = Constraints.orthogonality(hidden_states, mode: :spectral)

# Axis options for 3D tensors [batch, seq, vocab]
penalty = Constraints.orthogonality(logits, axis: :sequence)   # Decorrelate positions
penalty = Constraints.orthogonality(embeddings, axis: :vocabulary)  # Decorrelate dimensions
```

```elixir
# Consistency - penalize divergence between paired outputs
clean_output = model.(clean_input)
noisy_output = model.(add_noise.(clean_input))

# MSE (default)
penalty = Constraints.consistency(clean_output, noisy_output)

# L1 distance
penalty = Constraints.consistency(clean_output, noisy_output, metric: :l1)

# Cosine distance
penalty = Constraints.consistency(clean_output, noisy_output, metric: :cosine)

# Symmetric KL for log-probabilities
penalty = Constraints.consistency(logprobs1, logprobs2, metric: :kl)
```

## Polaris Integration

Gradient-level weight decay transforms (AdamW-style):

```elixir
alias NxPenalties.Integration.Polaris, as: PolarisIntegration

# Add L2 weight decay to any optimizer
optimizer =
  Polaris.Optimizers.adam(learning_rate: 0.001)
  |> PolarisIntegration.add_l2_decay(0.01)

# Add L1 decay for sparsity
optimizer =
  Polaris.Optimizers.sgd(learning_rate: 0.01)
  |> PolarisIntegration.add_l1_decay(0.001)

# Elastic Net decay (combined L1 + L2)
optimizer =
  Polaris.Optimizers.adam(learning_rate: 0.001)
  |> PolarisIntegration.add_elastic_net_decay(0.01, 0.3)  # 30% L1, 70% L2

# Compose multiple transforms
optimizer =
  Polaris.Optimizers.adam(learning_rate: 0.001)
  |> PolarisIntegration.add_l2_decay(0.01)
  |> PolarisIntegration.add_l1_decay(0.001)
```

**Loss-Based vs Gradient-Based**: Loss-based regularization (pipeline) adds penalty to loss before backprop. Gradient-based (Polaris transforms) modifies gradients directly. They're equivalent for SGD but differ for adaptive optimizers like Adam—gradient-based is generally preferred for modern training.

## Axon Integration

Wrap your loss function with regularization:

```elixir
alias NxPenalties.Integration.Axon, as: AxonIntegration

# Create penalty pipeline
pipeline = NxPenalties.pipeline([
  {:l2, weight: 0.01}
])

# Wrap loss function
regularized_loss = AxonIntegration.wrap_loss_with_pipeline(
  &Axon.Losses.mean_squared_error/2,
  pipeline
)

# Use in training
model
|> Axon.Loop.trainer(regularized_loss, optimizer)
|> Axon.Loop.run(data, epochs: 10)
```

## API Reference

### Penalty Functions

| Function | Description | Options |
|----------|-------------|---------|
| `l1/2` | L1 norm (Lasso) | `lambda`, `reduction` |
| `l2/2` | L2 norm squared (Ridge) | `lambda`, `reduction`, `center`, `clip` |
| `elastic_net/2` | Combined L1+L2 | `lambda`, `l1_ratio`, `reduction` |

### Divergence Functions

| Function | Description | Options |
|----------|-------------|---------|
| `kl_divergence/3` | KL(P \|\| Q) | `reduction` |
| `js_divergence/3` | Jensen-Shannon | `reduction` |
| `entropy/2` | Shannon entropy | `mode`, `reduction`, `normalize` |

### Gradient Penalty Functions

| Function | Description | Options |
|----------|-------------|---------|
| `gradient_penalty/3` | Gradient norm penalty (expensive) | `target_norm` |
| `output_magnitude_penalty/2` | Cheaper proxy for gradient penalty | `target`, `reduction` |
| `interpolated_gradient_penalty/4` | WGAN-GP style interpolated penalty | `target_norm` |

### Pipeline Functions

| Function | Description |
|----------|-------------|
| `pipeline/1` | Create pipeline from keyword list |
| `compute/3` | Execute pipeline, return `{total, metrics}` |
| `compute_total/3` | Execute pipeline, return tensor only (gradient-safe) |
| `Pipeline.add/4` | Add penalty to pipeline |
| `Pipeline.update_weight/3` | Change penalty weight |
| `Pipeline.set_enabled/3` | Enable/disable penalty |

### Constraint Functions

| Function | Description | Options |
|----------|-------------|---------|
| `Constraints.orthogonality/2` | Decorrelation penalty | `mode` (`:soft`/`:hard`/`:spectral`), `normalize`, `axis` (`:rows`/`:sequence`/`:vocabulary`) |
| `Constraints.consistency/3` | Paired output consistency | `metric` (`:mse`/`:l1`/`:cosine`/`:kl`), `reduction` |

### Polaris Transforms

| Function | Description | Parameters |
|----------|-------------|------------|
| `Integration.Polaris.add_l2_decay/2` | AdamW-style weight decay | `decay` (default: `0.01`) |
| `Integration.Polaris.add_l1_decay/2` | Sparsity-inducing decay | `decay` (default: `0.001`) |
| `Integration.Polaris.add_elastic_net_decay/3` | Combined L1+L2 decay | `decay`, `l1_ratio` |

### Utility Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `NxPenalties.validate/1` | Check for NaN/Inf | `{:ok, tensor}` or `{:error, :nan\|:inf}` |
| `GradientTracker.compute_grad_norm/2` | Gradient L2 norm | `float() \| nil` |
| `GradientTracker.pipeline_grad_norms/2` | Per-penalty grad norms | `map()` |
| `GradientTracker.total_grad_norm/2` | Total pipeline grad norm | `float() \| nil` |

## Telemetry Events

NxPenalties emits telemetry events for monitoring:

```elixir
# Attach handler
:telemetry.attach(
  "nx-penalties-logger",
  [:nx_penalties, :pipeline, :compute, :stop],
  fn _event, measurements, metadata, _config ->
    Logger.info("Pipeline computed in #{measurements.duration}ns")
    Logger.info("Metrics: #{inspect(metadata.metrics)}")
  end,
  nil
)
```

| Event | Measurements | Metadata |
|-------|-------------|----------|
| `[:nx_penalties, :pipeline, :compute, :start]` | `system_time` | `size` |
| `[:nx_penalties, :pipeline, :compute, :stop]` | `duration` | `metrics`, `total` |

## Debugging & Monitoring

### Gradient Tracking

Monitor which penalties contribute most to the gradient signal:

```elixir
pipeline = NxPenalties.pipeline([
  {:l1, weight: 0.001},
  {:l2, weight: 0.01},
  {:entropy, weight: 0.1, opts: [mode: :penalty]}
])

# Enable gradient norm tracking
{total, metrics} = NxPenalties.compute(pipeline, tensor, track_grad_norms: true)

metrics["l1_grad_norm"]       # L2 norm of L1 penalty's gradient
metrics["l2_grad_norm"]       # L2 norm of L2 penalty's gradient
metrics["entropy_grad_norm"]  # L2 norm of entropy penalty's gradient
metrics["total_grad_norm"]    # Combined gradient norm
```

**What it measures**: These are `∂penalty/∂(pipeline_input)`, not `∂penalty/∂params`. The "pipeline input" is whatever tensor you pass to `compute/3`—typically model outputs, activations, or logprobs.

**Performance note**: Gradient tracking requires additional backward passes. Only enable when debugging or for periodic monitoring (e.g., every 100 steps).

### Validation

Check tensors for numerical issues:

```elixir
case NxPenalties.validate(tensor) do
  {:ok, tensor} -> # Tensor is finite, proceed
  {:error, :nan} -> Logger.warning("NaN detected in tensor")
  {:error, :inf} -> Logger.warning("Inf detected in tensor")
end
```

## Performance

All penalty functions are implemented using `Nx.Defn` for JIT compilation:

- **GPU acceleration** - Automatically uses EXLA/CUDA when available
- **Fused operations** - Penalties compose efficiently in the computation graph
- **Minimal overhead** - No runtime option parsing in hot path

## Testing

```bash
# Run tests
mix test

# Run with coverage
mix coveralls.html

# Run quality checks
mix quality  # format + credo + dialyzer
```

## Examples

See the `examples/` directory for complete usage examples:

- `basic_usage.exs` - L1, L2, Elastic Net penalty functions
- `pipeline_composition.exs` - Pipeline creation and manipulation
- `curriculum_learning.exs` - Dynamic weight adjustment over epochs
- `axon_training.exs` - Axon neural network integration
- `polaris_integration.exs` - Gradient-level weight decay transforms
- `constraints.exs` - Orthogonality and consistency penalties
- `entropy_normalization.exs` - Entropy bonus/penalty with normalization
- `gradient_penalty.exs` - Gradient penalties and proxies
- `gradient_tracking.exs` - Monitoring gradient norms

Run examples with:

```bash
mix run examples/basic_usage.exs
./examples/run_all.sh  # Run all examples
```

## Notes

- NxPenalties is tensor-only. Data-aware adapters (e.g., selecting targets from `loss_fn_inputs`) live in downstream libraries such as Tinkex.
- Gradient penalties are computationally heavy; use sparingly and consider proxies.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit PRs to the `main` branch.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests first (TDD)
4. Ensure all checks pass (`mix quality && mix test`)
5. Submit a pull request

## License

MIT License - Copyright (c) 2025 North-Shore-AI

See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Nx](https://github.com/elixir-nx/nx) - Numerical computing for Elixir
- [Axon](https://github.com/elixir-nx/axon) - Neural network library
- [Polaris](https://github.com/elixir-nx/polaris) - Gradient optimization
