# ADR-006: Consistency Regularizer

## Status

Proposed

## Context

Consistency regularization encourages a model to produce similar outputs for similar inputs. This improves:

1. **Robustness** - stable under input perturbations
2. **Generalization** - smoother decision boundaries
3. **Semi-supervised learning** - unlabeled data provides consistency signal

Consistency can be measured between:
- Original and augmented versions of the same prompt
- Different samples from the same prompt (if using temperature > 0)
- Predictions at different training steps (temporal consistency)

## Decision

Implement consistency as a tensor primitive `NxPenalties.Constraints.consistency/3` (supports `:mse`, `:l1`, `:cosine`, `:kl` metrics) with a Tinkex adapter that resolves paired outputs from `loss_fn_inputs`.

### Interface

```elixir
# NxPenalties primitive (tensor-only)
loss = NxPenalties.Constraints.consistency(output1, output2,
  metric: :mse,   # or :l1/:cosine/:kl
  reduction: :mean
)

# Tinkex adapter (data-aware signature)
defmodule Tinkex.Regularizers.Consistency do
  @behaviour Tinkex.Regularizer

  @impl true
  def compute(data, logprobs, opts \\ []) do
    reference = resolve_reference!(data, Keyword.get(opts, :pair_field, "original_logprobs"))
    metric = Keyword.get(opts, :metric, :mse)
    reduction = Keyword.get(opts, :reduction, :mean)

    loss =
      NxPenalties.Constraints.consistency(
        logprobs,
        reference,
        metric: metric,
        reduction: reduction
      )

    {loss, %{"consistency_metric" => Atom.to_string(metric)}}
  end

  defp resolve_reference!(data, field) do
    data
    |> List.first()
    |> Map.get(:loss_fn_inputs, %{})
    |> Map.fetch!(field)
  end
end
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `:pair_field` | string | `"original_logprobs"` | Field containing reference logprobs (Tinkex adapter) |
| `:metric` | atom | `:mse` | Divergence metric (`:mse`, `:kl`, `:cosine`, `:l1`) |
| `:reduction` | `:mean` \| `:sum` \| `:none` | `:mean` | Aggregation method |

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
