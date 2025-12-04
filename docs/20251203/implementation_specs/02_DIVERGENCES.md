# 02: Divergences Implementation Specification

## Overview

This document specifies distribution-based penalties: KL divergence, Jensen-Shannon divergence, and entropy. These operate on probability distributions (typically as log-probabilities for numerical stability) and are critical for:

1. **Preventing catastrophic forgetting** - KL keeps fine-tuned model close to base
2. **Exploration in RL** - Entropy bonus prevents mode collapse
3. **Variational inference** - KL to prior in VAEs

## Module: NxPenalties.Divergences

### File Location
```
lib/nx_penalties/divergences.ex
```

### Module Structure
```elixir
defmodule NxPenalties.Divergences do
  @moduledoc """
  Distribution-based divergence and entropy penalties.

  All functions expect log-probabilities (logprobs) as input for numerical
  stability. Logprobs are in range (-∞, 0] where 0 = probability 1.

  ## Numerical Stability

  These functions implement standard stability tricks:
  - Log-sum-exp for entropy computation
  - Epsilon clamping to avoid log(0)
  - Proper handling of -inf values

  ## Common Shapes

  Functions expect logprobs with shape:
  - `{batch, vocab}` - per-token distributions
  - `{batch, seq, vocab}` - sequence of distributions
  - `{vocab}` - single distribution

  The last axis is assumed to be the vocabulary/distribution dimension.
  """

  import Nx.Defn

  @epsilon 1.0e-10
end
```

---

## Function: kl_divergence/3

### Specification

```elixir
@doc """
Kullback-Leibler divergence: KL(P || Q).

Measures how distribution P diverges from reference distribution Q.
Both inputs should be log-probabilities.

KL(P || Q) = Σ P(x) * (log P(x) - log Q(x))
           = Σ exp(log_p) * (log_p - log_q)

## Options

  * `:reduction` - How to reduce across batch/sequence. Default: `:mean`
    * `:mean` - Average KL across all positions
    * `:sum` - Sum KL across all positions
    * `:none` - Return per-position KL values
  * `:epsilon` - Small value added to prevent log(0). Default: `1.0e-10`

## Examples

    iex> p_logprobs = Nx.tensor([[-1.0, -2.0, -3.0]])  # batch=1, vocab=3
    iex> q_logprobs = Nx.tensor([[-1.5, -1.5, -1.5]])  # uniform-ish
    iex> NxPenalties.Divergences.kl_divergence(p_logprobs, q_logprobs)

## Notes

- KL(P||Q) ≠ KL(Q||P) - asymmetric!
- Forward KL (P||Q) is "mode-covering": P avoids zeros in Q
- For symmetric version, use `js_divergence/3`
"""
@spec kl_divergence(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
defn kl_divergence(p_logprobs, q_logprobs, opts \\ []) do
  reduction = opts[:reduction] || :mean

  # P(x) = exp(log_p)
  p = Nx.exp(p_logprobs)

  # KL = Σ p * (log_p - log_q)
  log_ratio = Nx.subtract(p_logprobs, q_logprobs)
  kl_pointwise = Nx.multiply(p, log_ratio)

  # Sum over distribution axis (last axis)
  kl_per_position = Nx.sum(kl_pointwise, axes: [-1])

  case reduction do
    :none -> kl_per_position
    :sum -> Nx.sum(kl_per_position)
    :mean -> Nx.mean(kl_per_position)
  end
end
```

### Implementation Notes

1. **Log-space Computation**: We work in log-space throughout to avoid underflow. `log_p - log_q = log(p/q)`.

2. **Handling -inf**: When `p = 0` (log_p = -inf), the contribution should be 0 (by convention 0 * log(0) = 0). Since `exp(-inf) = 0`, the multiply handles this correctly.

3. **Q = 0 Danger**: If `q = 0` but `p > 0`, KL = +inf. This is mathematically correct but may cause training issues. Consider clamping or warning.

4. **Shape Broadcasting**: P and Q must have the same shape. No implicit broadcasting on the distribution axis.

### Test Cases

```elixir
describe "kl_divergence/3" do
  test "identical distributions have zero KL" do
    logprobs = Nx.tensor([[-1.0, -2.0, -3.0]])
    result = NxPenalties.Divergences.kl_divergence(logprobs, logprobs)
    assert_close(result, Nx.tensor(0.0), atol: 1.0e-6)
  end

  test "uniform P to peaked Q has positive KL" do
    # P is uniform-ish, Q is peaked on first element
    p = Nx.tensor([[-1.1, -1.1, -1.1]])  # ~uniform
    q = Nx.tensor([[-0.1, -5.0, -5.0]])  # peaked
    result = NxPenalties.Divergences.kl_divergence(p, q)
    assert Nx.to_number(result) > 0
  end

  test "handles batch dimension" do
    p = Nx.tensor([[-1.0, -2.0], [-0.5, -1.5]])
    q = Nx.tensor([[-1.5, -1.5], [-1.0, -1.0]])
    result = NxPenalties.Divergences.kl_divergence(p, q, reduction: :none)
    assert Nx.shape(result) == {2}
  end

  test "sum reduction" do
    p = Nx.tensor([[-1.0, -2.0], [-0.5, -1.5]])
    q = Nx.tensor([[-1.5, -1.5], [-1.0, -1.0]])
    none = NxPenalties.Divergences.kl_divergence(p, q, reduction: :none)
    sum_result = NxPenalties.Divergences.kl_divergence(p, q, reduction: :sum)
    assert_close(sum_result, Nx.sum(none))
  end

  test "handles 3D input (batch, seq, vocab)" do
    p = Nx.random_uniform({2, 4, 100})
    p = Nx.subtract(p, Nx.logsumexp(p, axes: [-1], keep_axes: true))  # normalize
    q = Nx.random_uniform({2, 4, 100})
    q = Nx.subtract(q, Nx.logsumexp(q, axes: [-1], keep_axes: true))

    result = NxPenalties.Divergences.kl_divergence(p, q)
    assert Nx.shape(result) == {}  # scalar
  end

  test "gradient flows correctly" do
    grad_fn = Nx.Defn.grad(fn p ->
      q = Nx.tensor([[-1.0, -1.0, -1.0]])
      NxPenalties.Divergences.kl_divergence(p, q)
    end)
    p = Nx.tensor([[-0.5, -1.5, -2.5]])
    grads = grad_fn.(p)
    assert Nx.shape(grads) == {1, 3}
  end
end
```

---

## Function: js_divergence/3

### Specification

```elixir
@doc """
Jensen-Shannon divergence: symmetric KL.

JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
where M = 0.5 * (P + Q)

## Options

  * `:reduction` - How to reduce. Default: `:mean`
  * `:base` - Logarithm base for result scaling. Default: `nil` (natural log)
    * `:2` - Result in bits (0 to 1 range for probability distributions)

## Properties

  - Symmetric: JS(P, Q) = JS(Q, P)
  - Bounded: 0 ≤ JS ≤ log(2) ≈ 0.693 (or 1 bit with base 2)
  - Smooth: Always finite for valid distributions
"""
@spec js_divergence(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
defn js_divergence(p_logprobs, q_logprobs, opts \\ []) do
  reduction = opts[:reduction] || :mean

  # Compute M = 0.5 * (P + Q) in log space
  # log(0.5 * (exp(log_p) + exp(log_q))) = logsumexp([log_p, log_q]) - log(2)
  p = Nx.exp(p_logprobs)
  q = Nx.exp(q_logprobs)
  m = Nx.divide(Nx.add(p, q), 2)
  m_logprobs = Nx.log(Nx.max(m, @epsilon))

  # JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
  kl_pm = kl_divergence_internal(p_logprobs, m_logprobs)
  kl_qm = kl_divergence_internal(q_logprobs, m_logprobs)

  js_per_position = Nx.divide(Nx.add(kl_pm, kl_qm), 2)

  case reduction do
    :none -> js_per_position
    :sum -> Nx.sum(js_per_position)
    :mean -> Nx.mean(js_per_position)
  end
end

# Internal helper without reduction
defnp kl_divergence_internal(p_logprobs, q_logprobs) do
  p = Nx.exp(p_logprobs)
  log_ratio = Nx.subtract(p_logprobs, q_logprobs)
  kl_pointwise = Nx.multiply(p, log_ratio)
  Nx.sum(kl_pointwise, axes: [-1])
end
```

### Implementation Notes

1. **Numerical Stability**: Computing M in probability space then taking log is more stable than trying to do logsumexp tricks.

2. **Boundedness**: Unlike KL which can be infinite, JS is always bounded. This makes it safer for loss functions.

3. **Square Root**: Some papers use `sqrt(JS)` as a metric (Jensen-Shannon distance). Consider exposing as option.

### Test Cases

```elixir
describe "js_divergence/3" do
  test "symmetric: JS(P,Q) = JS(Q,P)" do
    p = Nx.tensor([[-1.0, -2.0, -3.0]])
    q = Nx.tensor([[-0.5, -1.5, -2.5]])
    js_pq = NxPenalties.Divergences.js_divergence(p, q)
    js_qp = NxPenalties.Divergences.js_divergence(q, p)
    assert_close(js_pq, js_qp)
  end

  test "identical distributions have zero JS" do
    logprobs = Nx.tensor([[-1.0, -2.0, -3.0]])
    result = NxPenalties.Divergences.js_divergence(logprobs, logprobs)
    assert_close(result, Nx.tensor(0.0), atol: 1.0e-6)
  end

  test "bounded by log(2)" do
    # Maximally different: one-hot distributions
    p = Nx.tensor([[0.0, -100.0, -100.0]])  # all mass on first
    q = Nx.tensor([[-100.0, 0.0, -100.0]])  # all mass on second
    result = NxPenalties.Divergences.js_divergence(p, q)
    assert Nx.to_number(result) <= :math.log(2) + 1.0e-6
  end
end
```

---

## Function: entropy/2

### Specification

```elixir
@doc """
Shannon entropy: H(P) = -Σ P(x) log P(x)

Measures uncertainty/spread of a distribution. Higher entropy means
more uniform; lower entropy means more peaked.

## Options

  * `:mode` - Direction of penalty. Default: `:penalty`
    * `:penalty` - Returns H (minimizing loss = minimize entropy = peaked)
    * `:bonus` - Returns -H (minimizing loss = maximize entropy = diverse)
  * `:reduction` - How to reduce. Default: `:mean`
  * `:normalize` - Normalize by max entropy (log vocab size). Default: `false`
    * When true, result is in range [0, 1]

## Examples

    # Encourage confident predictions (minimize entropy)
    entropy_penalty = NxPenalties.Divergences.entropy(logprobs, mode: :penalty)

    # Encourage diverse outputs (maximize entropy, RL exploration)
    entropy_bonus = NxPenalties.Divergences.entropy(logprobs, mode: :bonus)

## Notes

- Max entropy = log(vocab_size) for uniform distribution
- Entropy of one-hot = 0
"""
@spec entropy(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
defn entropy(logprobs, opts \\ []) do
  mode = opts[:mode] || :penalty
  reduction = opts[:reduction] || :mean
  normalize = opts[:normalize] || false

  # H = -Σ p * log(p) = -Σ exp(logp) * logp
  p = Nx.exp(logprobs)

  # Handle 0 * -inf = NaN case: where p=0, contribution should be 0
  # exp(-inf) = 0, and 0 * -inf gives NaN in some backends
  # Solution: mask out -inf before multiply
  safe_logprobs = Nx.select(
    Nx.less(logprobs, -100),  # treat < -100 as effectively 0
    Nx.tensor(0.0),
    logprobs
  )

  entropy_pointwise = Nx.negate(Nx.multiply(p, safe_logprobs))
  entropy_per_position = Nx.sum(entropy_pointwise, axes: [-1])

  # Optionally normalize
  entropy_per_position = if normalize do
    vocab_size = Nx.axis_size(logprobs, -1)
    max_entropy = Nx.log(Nx.tensor(vocab_size, type: Nx.type(logprobs)))
    Nx.divide(entropy_per_position, max_entropy)
  else
    entropy_per_position
  end

  reduced = case reduction do
    :none -> entropy_per_position
    :sum -> Nx.sum(entropy_per_position)
    :mean -> Nx.mean(entropy_per_position)
  end

  # Apply mode
  case mode do
    :penalty -> reduced
    :bonus -> Nx.negate(reduced)
  end
end
```

### Implementation Notes

1. **0 * log(0) Problem**: By L'Hôpital's rule, this limit is 0. We handle via masking very small values.

2. **Mode Semantics**:
   - `:penalty` - User wants to minimize entropy (peaked predictions)
   - `:bonus` - User wants to maximize entropy (add negative entropy to loss, so minimizing loss = maximizing entropy)

3. **Normalization**: Useful for comparing entropy across different vocab sizes or reporting as "utilization" metric.

### Test Cases

```elixir
describe "entropy/2" do
  test "uniform distribution has max entropy" do
    vocab_size = 4
    uniform_logprobs = Nx.broadcast(-:math.log(vocab_size), {1, vocab_size})
    result = NxPenalties.Divergences.entropy(uniform_logprobs, normalize: true)
    assert_close(result, Nx.tensor(1.0), atol: 1.0e-5)
  end

  test "one-hot distribution has zero entropy" do
    one_hot_logprobs = Nx.tensor([[0.0, -100.0, -100.0, -100.0]])
    result = NxPenalties.Divergences.entropy(one_hot_logprobs)
    assert_close(result, Nx.tensor(0.0), atol: 1.0e-5)
  end

  test "bonus mode negates" do
    logprobs = Nx.tensor([[-1.0, -1.5, -2.0]])
    penalty = NxPenalties.Divergences.entropy(logprobs, mode: :penalty)
    bonus = NxPenalties.Divergences.entropy(logprobs, mode: :bonus)
    assert_close(Nx.negate(penalty), bonus)
  end

  test "handles batch and sequence dimensions" do
    logprobs = Nx.random_uniform({2, 5, 100})
    logprobs = Nx.subtract(logprobs, Nx.logsumexp(logprobs, axes: [-1], keep_axes: true))

    none = NxPenalties.Divergences.entropy(logprobs, reduction: :none)
    assert Nx.shape(none) == {2, 5}

    mean = NxPenalties.Divergences.entropy(logprobs, reduction: :mean)
    assert Nx.shape(mean) == {}
  end

  test "gradient for entropy penalty encourages peakedness" do
    # Gradient should push probability toward less uniform
    grad_fn = Nx.Defn.grad(fn logp ->
      NxPenalties.Divergences.entropy(logp, mode: :penalty)
    end)
    # Near-uniform distribution
    logprobs = Nx.tensor([[-1.1, -1.1, -1.1]])
    grads = grad_fn.(logprobs)
    # All gradients should be similar (symmetric)
    assert Nx.shape(grads) == {1, 3}
  end
end
```

---

## Numerical Stability Module

### File Location
```
lib/nx_penalties/divergences/stability.ex
```

```elixir
defmodule NxPenalties.Divergences.Stability do
  @moduledoc """
  Numerical stability utilities for divergence computations.
  """

  import Nx.Defn

  @doc """
  Safe log that clamps input to avoid log(0) = -inf.
  """
  defn safe_log(x, epsilon \\ 1.0e-10) do
    Nx.log(Nx.max(x, epsilon))
  end

  @doc """
  Normalize logprobs to valid log-probability distribution.
  Ensures values sum to 1 (in probability space).
  """
  defn normalize_logprobs(logprobs) do
    Nx.subtract(logprobs, Nx.logsumexp(logprobs, axes: [-1], keep_axes: true))
  end

  @doc """
  Check if logprobs are valid (not NaN, not all -inf).
  Returns boolean tensor.
  """
  defn valid_logprobs?(logprobs) do
    has_nan = Nx.any(Nx.is_nan(logprobs))
    all_neg_inf = Nx.all(Nx.equal(logprobs, Nx.Constants.neg_infinity()))
    Nx.logical_not(Nx.logical_or(has_nan, all_neg_inf))
  end
end
```

---

## Usage Examples

### KL Regularization for Fine-tuning

```elixir
defmodule MyTraining do
  import Nx.Defn

  defn training_loss(y_true, y_pred, base_model_logprobs, opts) do
    # Task loss
    task_loss = Axon.Losses.categorical_cross_entropy(y_true, y_pred)

    # KL regularization to base model
    kl_penalty = NxPenalties.Divergences.kl_divergence(y_pred, base_model_logprobs)

    # Combine
    lambda_kl = opts[:lambda_kl] || 0.1
    Nx.add(task_loss, Nx.multiply(kl_penalty, lambda_kl))
  end
end
```

### Entropy Bonus for RL

```elixir
defmodule PolicyGradient do
  import Nx.Defn

  defn policy_loss(log_probs, advantages, opts) do
    # Policy gradient loss
    pg_loss = Nx.negate(Nx.mean(Nx.multiply(log_probs, advantages)))

    # Entropy bonus for exploration
    entropy_coef = opts[:entropy_coef] || 0.01
    entropy_bonus = NxPenalties.Divergences.entropy(log_probs, mode: :bonus)

    # Lower loss = higher entropy (more exploration)
    Nx.add(pg_loss, Nx.multiply(entropy_bonus, entropy_coef))
  end
end
```

---

## Performance Considerations

### Memory

| Function | Memory Pattern |
|----------|----------------|
| `kl_divergence/3` | O(input_size) - one intermediate tensor |
| `js_divergence/3` | O(3 * input_size) - M tensor + two KL calls |
| `entropy/2` | O(input_size) - p and masked logprobs |

### Optimization

1. **Fused Operations**: KL and entropy can be computed in single pass if backend supports fusion.

2. **Avoid Recomputation**: If computing both KL and entropy, extract shared p=exp(logprobs) computation.

3. **Large Vocab**: For large vocabularies (100k+), consider chunked computation to manage memory.

---

## Integration Checklist

- [ ] All functions accept log-probabilities (not raw probabilities)
- [ ] Clear documentation on input shapes and axes
- [ ] Numerical stability tests with edge cases
- [ ] Mode/direction options clearly documented
- [ ] Gradient tests for each function
- [ ] Comparison tests against reference implementations (NumPy/PyTorch)
