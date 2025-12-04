# ADR-007: Gradient Penalty Regularizer

## Status

**Proposed - Advanced/v0.2**

> ⚠️ **Implementation Note:** This regularizer requires second-order derivatives (gradient of gradient), which is computationally expensive. For v0.1, consider using `output_magnitude_penalty` as a simpler proxy. Full WGAN-GP style gradient penalty is planned for v0.2 with explicit performance warnings.

## Context

Gradient penalty regularization constrains the norm of gradients, encouraging smoother functions. Originally popularized by WGAN-GP for training GANs, gradient penalties have broader applications:

1. **Lipschitz constraint** - bounds how fast outputs can change
2. **Training stability** - prevents exploding gradients
3. **Adversarial robustness** - limits sensitivity to input perturbations

In Tinker's context, we can apply gradient penalties to the relationship between inputs and logprobs, encouraging the fine-tuned model to have bounded sensitivity.

## Decision

Implement gradient penalties via `NxPenalties.GradientPenalty` primitives, with a Tinkex adapter to fit the data/logprobs regularizer contract.

### Interface

```elixir
# Tensor primitives (NxPenalties)
gp = NxPenalties.GradientPenalty.gradient_penalty(loss_fn, logprobs, target_norm: 1.0)
interp_gp =
  NxPenalties.GradientPenalty.interpolated_gradient_penalty(
    loss_fn,
    fake_logprobs,
    real_logprobs,
    target_norm: 1.0
  )
proxy = NxPenalties.GradientPenalty.output_magnitude_penalty(logprobs, target: 1.0)

# Tinkex adapter (data-aware signature)
defmodule Tinkex.Regularizers.GradientPenalty do
  @behaviour Tinkex.Regularizer

  @impl true
  def compute(data, logprobs, opts \\ []) do
    mode = Keyword.get(opts, :mode, :output)
    loss_fn = Keyword.get(opts, :loss_fn, fn x -> Nx.sum(x) end)

    case mode do
      :output ->
        penalty = NxPenalties.GradientPenalty.gradient_penalty(loss_fn, logprobs, opts)
        {penalty, %{}}

      :interpolated ->
        reference =
          data
          |> List.first()
          |> Map.get(:loss_fn_inputs, %{})
          |> Map.fetch!(Keyword.get(opts, :reference_field, :reference_logprobs))

        penalty =
          NxPenalties.GradientPenalty.interpolated_gradient_penalty(
            loss_fn,
            logprobs,
            reference,
            opts
          )

        {penalty, %{}}
    end
  end
end
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `:target_norm` | float | `1.0` | Target gradient norm (NxPenalties) |
| `:mode` | atom | `:output` | Gradient computation mode (adapter) |
| `:reference_field` | string/atom | `"reference_logprobs"` | For interpolated mode (adapter) |
| `:loss_fn` | function | `fn x -> Nx.sum(x) end` | Function whose gradient is penalized (adapter) |

## Consequences

### Positive

- Encourages smooth, well-behaved model
- Improves training stability
- Can improve adversarial robustness
- Theoretically grounded (Lipschitz continuity)

### Negative

- Computationally expensive (requires gradient computation)
- Complex interaction with other training dynamics
- May slow down legitimate adaptation
- Requires careful weight tuning

### Neutral

- Target norm is configurable
- Multiple modes for different use cases

## Implementation Notes

### Computational Cost

Gradient penalty requires computing second-order derivatives (gradient of gradient). This approximately doubles the backward pass cost. Consider:

```elixir
# Use sparingly - maybe every N steps
if rem(step, 10) == 0 do
  include_gradient_penalty()
end
```

### Lipschitz Interpretation

A function f is K-Lipschitz if ||f(x) - f(y)|| ≤ K||x - y|| for all x, y.

Equivalently: ||∇f|| ≤ K everywhere.

Setting `target_norm: 1.0` encourages 1-Lipschitz behavior.

### Relationship to Weight Clipping

WGAN originally used weight clipping for Lipschitz constraint. Gradient penalty is a softer alternative that:
- Doesn't hard-limit capacity
- Provides smoother training signal
- Allows local violations

### Typical Weight Range

| Weight | Effect |
|--------|--------|
| 1.0 | Light smoothness encouragement |
| 10.0 | Standard WGAN-GP weight |
| 100.0 | Very strong Lipschitz enforcement |

The original WGAN-GP paper uses λ=10.

### Mode Selection

| Mode | Use Case |
|------|----------|
| `:output_wrt_logprobs` | General smoothness, no reference needed |
| `:interpolated` | When you have paired distributions (like KL regularizer) |

## Alternatives Considered

### 1. Spectral Normalization
Normalizes weights to have spectral norm 1. Requires weight access, not available in Tinker API.

### 2. Double Backprop
Older technique for gradient regularization. More complex, similar cost.

### 3. Input Gradient Regularization
Penalize ∂loss/∂input. More directly tied to adversarial robustness. Could be future variant.

### 4. Gradient Clipping
Hard clip gradients instead of soft penalty. Handled by optimizer, not regularizer.

## References

- Gulrajani et al. (2017). "Improved Training of Wasserstein GANs" (WGAN-GP)
- Miyato et al. (2018). "Spectral Normalization for Generative Adversarial Networks"
- Drucker & LeCun (1992). "Double Backpropagation"
