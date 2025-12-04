# ADR-005: Entropy Regularizer

## Status

Proposed

## Context

Entropy measures the uncertainty or "spread" of a probability distribution. In language model fine-tuning, entropy regularization serves two opposing purposes:

1. **Entropy maximization** - encourages diverse, uncertain outputs (exploration)
2. **Entropy minimization** - encourages confident, peaked outputs (exploitation)

The choice depends on the task:
- Creative generation benefits from higher entropy
- Classification/factual tasks benefit from lower entropy (confidence)
- RLHF often uses entropy bonus to prevent mode collapse

## Decision

Implement entropy regularization as a tensor primitive in `NxPenalties.Divergences.entropy/2` (configurable direction, normalization) and a Tinkex adapter to fit the `compute/3` contract.

### Interface

```elixir
# Tensor primitive (NxPenalties)
entropy = NxPenalties.Divergences.entropy(logprobs,
  mode: :bonus,   # or :penalty
  reduction: :mean, # or :sum/:none
  normalize: false
)

# Tinkex adapter (data-aware signature)
defmodule Tinkex.Regularizers.Entropy do
  @behaviour Tinkex.Regularizer

  @impl true
  def compute(_data, logprobs, opts \\ []) do
    mode =
      case Keyword.get(opts, :mode, :maximize) do
        :maximize -> :bonus
        :minimize -> :penalty
        other -> other
      end

    entropy =
      NxPenalties.Divergences.entropy(logprobs,
        mode: mode,
        normalize: Keyword.get(opts, :normalize, false),
        reduction: Keyword.get(opts, :reduction, :mean)
      )

    {entropy, %{
      "entropy" => Nx.to_number(entropy),
      "mode" => Atom.to_string(mode)
    }}
  end
end
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `:mode` | `:bonus`/`:penalty` (NxPenalties) or `:maximize`/`:minimize` (adapter) | `:maximize` | Direction of regularization |
| `:normalize` | boolean | `false` | Normalize by max possible entropy |
| `:reduction` | `:mean` \| `:sum` \| `:none` | `:mean` | Aggregation |

> **Future extension:** `:temperature` for distribution sharpening is documented below but not yet implemented in `compute/3`.

## Consequences

### Positive

- Simple, well-understood information-theoretic measure
- Directly controls output diversity/confidence tradeoff
- Prevents mode collapse when maximizing
- Encourages calibration when minimizing
- Normalized mode allows comparison across vocab sizes

### Negative

- Choosing direction requires understanding task requirements
- Entropy maximization can hurt task performance if overdone
- Doesn't directly measure output quality

### Neutral

- Returns statistics for monitoring entropy during training
- Normalization is optional but aids interpretability

## Implementation Notes

### Entropy Interpretation

| Entropy Level | Meaning | Normalized (32k vocab) |
|---------------|---------|------------------------|
| 0 | Deterministic (one token) | 0.0 |
| 2.3 | ~10 equally likely tokens | ~0.22 |
| 4.6 | ~100 equally likely tokens | ~0.44 |
| 10.4 | Max (uniform over 32k) | 1.0 |

### Numerical Stability

When p ≈ 0, p * log(p) → 0 (L'Hôpital's rule). Nx handles this correctly, but we can add explicit safety:

```elixir
# Clip logprobs to prevent -inf * 0 = NaN
logprobs_safe = Nx.clip(logprobs, -100, 0)
```

### Temperature Scaling

Optional temperature parameter adjusts distribution sharpness before entropy computation:

```elixir
if temperature != 1.0 do
  logprobs = Nx.divide(logprobs, temperature)
  # Re-normalize with log_softmax
  logprobs = Nx.subtract(logprobs, Nx.logsumexp(logprobs, axes: [-1], keep_axes: true))
end
```

### Relationship to RLHF

In Proximal Policy Optimization (PPO), entropy bonus is standard:

```
Loss = -reward + β * KL_penalty - γ * entropy_bonus
```

This regularizer provides the entropy bonus term. Combined with ADR-004 (KL Divergence), you can implement PPO-style objectives.

### Typical Weight Range

| Weight | Effect | Use Case |
|--------|--------|----------|
| 0.001 | Very light | Minor diversity encouragement |
| 0.01 | Moderate | Standard entropy bonus |
| 0.1 | Strong | Aggressive exploration |
| 1.0+ | Dominant | Entropy becomes primary objective |

## Alternatives Considered

### 1. Conditional Entropy
H(Y|X) for sequence models. More complex, requires positional awareness. Could be future ADR.

### 2. Mutual Information
I(X;Y) = H(X) - H(X|Y). Requires paired distributions. Out of scope.

### 3. Rényi Entropy
Generalization with parameter α. More flexible but less common. Could add if needed.

### 4. Top-k Entropy
Only compute entropy over top-k tokens. Reduces noise from low-probability tokens.

```elixir
# Potential enhancement
topk_logprobs = Nx.top_k(logprobs, k: 100)
# Compute entropy only over these
```

## References

- Shannon (1948). "A Mathematical Theory of Communication"
- Williams & Peng (1991). "Function Optimization using Connectionist Reinforcement Learning Algorithms" (entropy bonus in RL)
- Mnih et al. (2016). "Asynchronous Methods for Deep Reinforcement Learning" (A3C entropy term)
