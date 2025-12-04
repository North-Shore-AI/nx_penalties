# ADR-004: KL Divergence Regularizer

## Status

Proposed

## Context

Kullback-Leibler divergence measures how one probability distribution differs from a reference distribution. In fine-tuning, KL regularization is critical for:

1. **Preventing catastrophic forgetting** - keeping fine-tuned model close to base model
2. **Maintaining capabilities** - base model's general knowledge preserved
3. **Controlled adaptation** - fine-tune for task while preserving base behavior

This is arguably the most important regularizer for LoRA fine-tuning, as it directly addresses the core challenge of adapting a model without destroying it.

## Decision

Implement `Tinkex.Regularizer.KLDivergence` with support for comparing against reference logprobs.

### Interface

```elixir
defmodule Tinkex.Regularizer.KLDivergence do
  @behaviour Tinkex.Regularizer

  @moduledoc """
  KL Divergence regularizer for keeping fine-tuned outputs close to reference.

  KL(P || Q) = Σ P(x) * log(P(x) / Q(x))

  Where P is the fine-tuned distribution and Q is the reference (base model).

  ## Usage

  Requires reference logprobs to be provided either:
  1. Via opts[:reference_logprobs] - pre-computed Nx tensor
  2. Via opts[:reference_field] - field name in Datum.loss_fn_inputs
  3. Via opts[:compute_reference] - function to compute reference

  ## Example

      # Pre-compute base model logprobs
      {:ok, base_response} = SamplingClient.compute_logprobs(base_client, prompt)
      base_logprobs = base_response.logprobs

      # Use in regularizer
      %RegularizerSpec{
        fn: &KLDivergence.compute/3,
        weight: 0.1,
        name: "kl_div",
        opts: [reference_logprobs: base_logprobs]
      }
  """

  @impl true
  def compute(data, logprobs, opts \\ []) do
    reference = get_reference(data, opts)

    unless reference do
      raise ArgumentError, """
      KLDivergence requires reference logprobs. Provide one of:
        - reference_logprobs: Nx.Tensor.t()
        - reference_field: atom (field in Datum.loss_fn_inputs)
        - compute_reference: (data -> Nx.Tensor.t())
      """
    end

    # Ensure same shape
    {logprobs, reference} = broadcast_shapes(logprobs, reference)

    # Convert logprobs to probabilities
    p = Nx.exp(logprobs)       # fine-tuned distribution
    q = Nx.exp(reference)       # reference distribution

    # KL(P || Q) = Σ p * log(p/q) = Σ p * (log_p - log_q)
    # Using logprobs directly for numerical stability
    kl_pointwise = Nx.multiply(p, Nx.subtract(logprobs, reference))

    # Sum over vocabulary, mean over sequence/batch
    kl_per_position = Nx.sum(kl_pointwise, axes: [-1])
    kl_value = Nx.mean(kl_per_position)

    {kl_value, %{
      "kl_divergence" => Nx.to_number(kl_value),
      "kl_max" => Nx.to_number(Nx.reduce_max(kl_per_position)),
      "kl_min" => Nx.to_number(Nx.reduce_min(kl_per_position))
    }}
  end

  @impl true
  def name, do: "kl_divergence"

  # Private helpers

  defp get_reference(data, opts) do
    cond do
      opts[:reference_logprobs] ->
        opts[:reference_logprobs]

      opts[:reference_field] ->
        # Extract from first datum's loss_fn_inputs
        field = opts[:reference_field]
        data
        |> List.first()
        |> Map.get(:loss_fn_inputs, %{})
        |> Map.get(field)
        |> maybe_to_tensor()

      opts[:compute_reference] ->
        opts[:compute_reference].(data)

      true ->
        nil
    end
  end

  defp broadcast_shapes(a, b) do
    # Handle shape mismatches gracefully
    {a, b}
  end

  defp maybe_to_tensor(%Tinkex.Types.TensorData{} = td), do: Tinkex.Types.TensorData.to_nx(td)
  defp maybe_to_tensor(%Nx.Tensor{} = t), do: t
  defp maybe_to_tensor(nil), do: nil
end
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `:reference_logprobs` | Nx.Tensor.t() | required | Reference distribution logprobs |
| `:reference_field` | atom | nil | Field name in Datum.loss_fn_inputs |
| `:compute_reference` | function | nil | Function to compute reference on-demand |
| `:direction` | `:forward` \| `:reverse` | `:forward` | KL(P\|\|Q) vs KL(Q\|\|P) |
| `:symmetric` | boolean | `false` | Use Jensen-Shannon divergence instead |

## Consequences

### Positive

- Directly addresses catastrophic forgetting
- Well-understood theoretically (information theory)
- Composable with other regularizers
- Multiple ways to provide reference distribution

### Negative

- Requires reference logprobs (extra API call or storage)
- Asymmetric (KL(P||Q) ≠ KL(Q||P))
- Can be numerically unstable with very different distributions
- Additional complexity in training pipeline

### Neutral

- Symmetric variant (Jensen-Shannon) available via option
- Returns distribution statistics for monitoring

## Implementation Notes

### Obtaining Reference Logprobs

Three patterns for getting base model logprobs:

```elixir
# Pattern 1: Pre-compute and cache
base_logprobs = precompute_base_logprobs(prompts)
opts = [reference_logprobs: base_logprobs]

# Pattern 2: Store in training data
datum = %Datum{
  model_input: input,
  loss_fn_inputs: %{"base_logprobs" => base_logprobs_tensor}
}
opts = [reference_field: "base_logprobs"]

# Pattern 3: Compute on-demand (expensive!)
opts = [compute_reference: fn data ->
  # Call base model - adds latency!
  compute_base_logprobs(data)
end]
```

Pattern 1 or 2 recommended for performance.

### Numerical Stability

KL divergence can blow up when P assigns probability to events Q considers impossible. Mitigations:

```elixir
# Add small epsilon to prevent log(0)
epsilon = 1.0e-10
q_safe = Nx.max(q, epsilon)

# Or clip extreme KL values
kl_clipped = Nx.clip(kl_pointwise, 0, 100)
```

### Typical Weight Range

KL regularization weights are typically 0.01 to 1.0, depending on how much drift from base model is acceptable:

| Weight | Effect |
|--------|--------|
| 0.01 | Light regularization, allows significant adaptation |
| 0.1 | Moderate, balanced adaptation/preservation |
| 0.5 | Strong, prioritizes base model behavior |
| 1.0+ | Very strong, minimal adaptation |

### Reverse KL vs Forward KL

- **Forward KL (P||Q)**: Mode-covering, fine-tuned model avoids zeros in reference
- **Reverse KL (Q||P)**: Mode-seeking, fine-tuned model can ignore some reference modes

Forward KL is default as it's safer for preserving capabilities.

## Alternatives Considered

### 1. Jensen-Shannon Divergence
Symmetric version: JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q).
Included as option via `:symmetric => true`.

### 2. Wasserstein Distance
More robust to distribution mismatch but computationally expensive. Could be separate ADR.

### 3. Maximum Mean Discrepancy (MMD)
Kernel-based distribution comparison. More complex, potential future ADR.

### 4. Simple MSE on logprobs
Crude approximation to KL. Simpler but less principled. Users can implement via custom fn.

## References

- Kullback & Leibler (1951). "On Information and Sufficiency"
- Hinton et al. (2015). "Distilling the Knowledge in a Neural Network"
- OpenAI (2022). "InstructGPT" - uses KL penalty in RLHF
