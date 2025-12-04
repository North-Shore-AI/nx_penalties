# ADR-006: Consistency Regularizer

## Status

Proposed

## Context

Consistency regularization encourages a model to produce similar outputs for similar inputs. This improves:

1. **Robustness** - stable under input perturbations
2. **Generalization** - smoother decision boundaries
3. **Semi-supervised learning** - unlabeled data provides consistency signal

In the context of Tinker, consistency can be measured between:
- Original and augmented versions of the same prompt
- Different samples from the same prompt (if using temperature > 0)
- Predictions at different training steps (temporal consistency)

## Decision

Implement `Tinkex.Regularizer.Consistency` for penalizing output divergence between paired inputs.

### Interface

```elixir
defmodule Tinkex.Regularizer.Consistency do
  @behaviour Tinkex.Regularizer

  @moduledoc """
  Consistency regularizer for encouraging stable outputs.

  Requires paired data: original and augmented/alternative inputs.
  Penalizes divergence between their output distributions.

  ## Data Format

  Expects Datum pairs where `loss_fn_inputs` contains:
  - `"original_logprobs"` or similar field for first view
  - Current forward pass provides second view

  Alternatively, use `:pair_field` to specify the reference.

  ## Example

      # Training data includes augmented view logprobs
      datum = %Datum{
        model_input: augmented_input,
        loss_fn_inputs: %{
          "original_logprobs" => original_logprobs
        }
      }

      %RegularizerSpec{
        fn: &Consistency.compute/3,
        weight: 1.0,
        name: "consistency",
        opts: [pair_field: "original_logprobs", metric: :mse]
      }
  """

  @impl true
  def compute(data, logprobs, opts \\ []) do
    pair_field = Keyword.get(opts, :pair_field, "original_logprobs")
    metric = Keyword.get(opts, :metric, :mse)

    # Extract reference logprobs from data
    reference = extract_reference(data, pair_field)

    unless reference do
      raise ArgumentError, """
      Consistency regularizer requires paired data.
      Provide reference logprobs via loss_fn_inputs["#{pair_field}"]
      """
    end

    # Compute divergence metric
    {loss, metrics} = compute_divergence(logprobs, reference, metric)

    {loss, Map.merge(metrics, %{"consistency_metric" => Atom.to_string(metric)})}
  end

  @impl true
  def name, do: "consistency"

  # Private helpers

  defp extract_reference(data, field) do
    data
    |> List.first()
    |> Map.get(:loss_fn_inputs, %{})
    |> Map.get(field)
    |> maybe_to_tensor()
  end

  defp compute_divergence(logprobs, reference, metric) do
    case metric do
      :mse ->
        # Mean squared error on logprobs
        diff = Nx.subtract(logprobs, reference)
        squared = Nx.power(diff, 2)
        mse = Nx.mean(squared)
        {mse, %{"mse" => Nx.to_number(mse)}}

      :kl ->
        # Symmetric KL (Jensen-Shannon style)
        p = Nx.exp(logprobs)
        q = Nx.exp(reference)

        kl_pq = Nx.sum(Nx.multiply(p, Nx.subtract(logprobs, reference)), axes: [-1])
        kl_qp = Nx.sum(Nx.multiply(q, Nx.subtract(reference, logprobs)), axes: [-1])

        js = Nx.mean(Nx.add(kl_pq, kl_qp)) |> Nx.divide(2)
        {js, %{"js_divergence" => Nx.to_number(js)}}

      :cosine ->
        # Cosine distance (1 - cosine similarity)
        # Flatten to vectors
        p_flat = Nx.reshape(logprobs, {:auto})
        q_flat = Nx.reshape(reference, {:auto})

        dot = Nx.sum(Nx.multiply(p_flat, q_flat))
        norm_p = Nx.sqrt(Nx.sum(Nx.power(p_flat, 2)))
        norm_q = Nx.sqrt(Nx.sum(Nx.power(q_flat, 2)))

        cosine_sim = Nx.divide(dot, Nx.multiply(norm_p, norm_q))
        cosine_dist = Nx.subtract(1, cosine_sim)

        {cosine_dist, %{
          "cosine_similarity" => Nx.to_number(cosine_sim),
          "cosine_distance" => Nx.to_number(cosine_dist)
        }}

      :l1 ->
        # L1 distance
        diff = Nx.abs(Nx.subtract(logprobs, reference))
        l1 = Nx.mean(diff)
        {l1, %{"l1_distance" => Nx.to_number(l1)}}
    end
  end

  defp maybe_to_tensor(%Tinkex.Types.TensorData{} = td), do: Tinkex.Types.TensorData.to_nx(td)
  defp maybe_to_tensor(%Nx.Tensor{} = t), do: t
  defp maybe_to_tensor(nil), do: nil
end
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `:pair_field` | string | `"original_logprobs"` | Field containing reference logprobs |
| `:metric` | atom | `:mse` | Divergence metric (`:mse`, `:kl`, `:cosine`, `:l1`) |

> **Future extension:** `:temperature` for distribution sharpening before comparison is planned but not yet implemented.

## Consequences

### Positive

- Improves robustness to input variations
- Enables semi-supervised learning patterns
- Multiple divergence metrics for different use cases
- Encourages smoother, more generalizable models

### Negative

- Requires paired data (original + augmented)
- Adds complexity to data preparation pipeline
- May conflict with task-specific adaptation goals
- Extra forward pass needed to get reference logprobs

### Neutral

- Choice of metric affects training dynamics
- Can be combined with other regularizers

## Implementation Notes

### Data Preparation Patterns

```elixir
# Pattern 1: Augment prompts, store original logprobs
original_logprobs = compute_logprobs(model, original_prompt)
augmented_prompt = augment(original_prompt)  # typos, paraphrase, etc.

datum = %Datum{
  model_input: ModelInput.from_text(augmented_prompt),
  loss_fn_inputs: %{"original_logprobs" => original_logprobs}
}

# Pattern 2: Multiple samples from same prompt
samples = for _ <- 1..N do
  sample(model, prompt, temperature: 0.7)
end
# Use first as reference, others in consistency loss
```

### Metric Selection Guide

| Metric | Properties | Best For |
|--------|-----------|----------|
| `:mse` | Symmetric, smooth gradients | General use, similar scales |
| `:kl` | Asymmetric awareness | Probability distributions |
| `:cosine` | Scale-invariant | Different magnitude distributions |
| `:l1` | Robust to outliers | When MSE is too sensitive |

### Augmentation Strategies

Common augmentations for text consistency:
1. **Synonym replacement** - swap words with synonyms
2. **Back-translation** - translate to another language and back
3. **Typo injection** - simulate user input errors
4. **Paraphrasing** - rephrase with same meaning
5. **Noise injection** - add/remove random tokens

### Typical Weight Range

| Weight | Effect |
|--------|--------|
| 0.1 | Light consistency pressure |
| 1.0 | Moderate, balanced with task loss |
| 10.0 | Strong, consistency-focused training |

## Alternatives Considered

### 1. Temporal Consistency (EMA)
Compare current predictions to exponential moving average of past predictions. Requires stateful regularizer.

### 2. Dropout Consistency
Run same input twice with different dropout masks. Not applicable to Tinker (no dropout control).

### 3. Mixup Consistency
Interpolate inputs and require interpolated outputs. Complex for discrete text.

## References

- Sajjadi et al. (2016). "Regularization With Stochastic Transformations and Perturbations for Deep Semi-Supervised Learning"
- Miyato et al. (2018). "Virtual Adversarial Training"
- Xie et al. (2020). "Unsupervised Data Augmentation for Consistency Training"
