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

Implement `Tinkex.Regularizer.GradientPenalty` for penalizing large gradient norms.

### Interface

```elixir
defmodule Tinkex.Regularizer.GradientPenalty do
  @behaviour Tinkex.Regularizer

  @moduledoc """
  Gradient penalty regularizer for Lipschitz smoothness.

  Penalizes ||∇_x f(x)||² to encourage bounded gradients.

  ## Modes

  - `:output_wrt_logprobs` - gradient of loss w.r.t. logprobs (default)
  - `:interpolated` - WGAN-GP style on interpolated points

  ## Note

  This regularizer uses Nx.Defn.grad internally, which may add
  computational overhead. Consider using sparingly or with
  lower weight.

  ## Example

      %RegularizerSpec{
        fn: &GradientPenalty.compute/3,
        weight: 10.0,
        name: "gradient_penalty",
        opts: [target_norm: 1.0]
      }
  """

  import Nx.Defn

  @impl true
  def compute(data, logprobs, opts \\ []) do
    target_norm = Keyword.get(opts, :target_norm, 1.0)
    mode = Keyword.get(opts, :mode, :output_wrt_logprobs)

    case mode do
      :output_wrt_logprobs ->
        compute_logprob_gradient_penalty(logprobs, target_norm)

      :interpolated ->
        reference = get_reference(data, opts)
        compute_interpolated_penalty(logprobs, reference, target_norm)
    end
  end

  @impl true
  def name, do: "gradient_penalty"

  # Compute gradient penalty on logprobs directly
  defp compute_logprob_gradient_penalty(logprobs, target_norm) do
    # Define a simple function whose gradient we'll penalize
    # Using sum of logprobs as proxy for "output"
    grad_fn = fn lp -> Nx.sum(lp) end

    # Compute gradient
    grad = Nx.Defn.grad(grad_fn).(logprobs)

    # Compute gradient norm
    grad_norm = compute_norm(grad)

    # Penalty: (||grad|| - target)²
    deviation = Nx.subtract(grad_norm, target_norm)
    penalty = Nx.power(deviation, 2)

    {penalty, %{
      "gradient_norm" => Nx.to_number(grad_norm),
      "target_norm" => target_norm,
      "penalty" => Nx.to_number(penalty)
    }}
  end

  # WGAN-GP style: interpolate between real and generated, penalize gradient
  defp compute_interpolated_penalty(logprobs, reference, target_norm) do
    # Random interpolation coefficient
    epsilon = Nx.random_uniform(Nx.shape(logprobs), type: Nx.type(logprobs))

    # Interpolated points
    interpolated = Nx.add(
      Nx.multiply(epsilon, logprobs),
      Nx.multiply(Nx.subtract(1, epsilon), reference)
    )

    # Compute gradient at interpolated points
    grad_fn = fn x -> Nx.sum(x) end
    grad = Nx.Defn.grad(grad_fn).(interpolated)

    # Gradient norm per sample
    grad_norm = compute_norm(grad)

    # WGAN-GP penalty: (||grad|| - 1)²
    deviation = Nx.subtract(grad_norm, target_norm)
    penalty = Nx.mean(Nx.power(deviation, 2))

    {penalty, %{
      "gradient_norm_mean" => Nx.to_number(Nx.mean(grad_norm)),
      "gradient_norm_max" => Nx.to_number(Nx.reduce_max(grad_norm)),
      "target_norm" => target_norm,
      "penalty" => Nx.to_number(penalty)
    }}
  end

  defp compute_norm(tensor) do
    # L2 norm: sqrt(sum(x²))
    squared = Nx.power(tensor, 2)
    Nx.sqrt(Nx.sum(squared))
  end

  defp get_reference(data, opts) do
    field = Keyword.get(opts, :reference_field, "reference_logprobs")
    data
    |> List.first()
    |> Map.get(:loss_fn_inputs, %{})
    |> Map.get(field)
    |> maybe_to_tensor()
  end

  defp maybe_to_tensor(%Tinkex.Types.TensorData{} = td), do: Tinkex.Types.TensorData.to_nx(td)
  defp maybe_to_tensor(%Nx.Tensor{} = t), do: t
  defp maybe_to_tensor(nil), do: nil
end
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `:target_norm` | float | `1.0` | Target gradient norm (1.0 for 1-Lipschitz) |
| `:mode` | atom | `:output_wrt_logprobs` | Gradient computation mode |
| `:reference_field` | string | `"reference_logprobs"` | For interpolated mode |

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
