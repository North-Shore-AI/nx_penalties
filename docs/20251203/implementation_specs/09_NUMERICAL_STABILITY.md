# 09: Numerical Stability Implementation Specification

## Overview

Numerical stability is critical for ML loss functions. This document specifies stability patterns, edge cases, and their handling across all NxPenalties functions.

## Common Numerical Issues

| Issue | Cause | Impact | Solution |
|-------|-------|--------|----------|
| **Underflow** | Very small values → 0 | Division by zero, log(0) | Epsilon clamping |
| **Overflow** | Very large values → Inf | Gradient explosion | Clipping, log-space |
| **NaN** | 0/0, Inf-Inf, etc. | Training crash | Input validation, safe ops |
| **Precision loss** | Subtracting similar values | Incorrect gradients | Kahan summation, log-space |

## Stability Patterns

### Pattern 1: Epsilon Clamping

```elixir
@epsilon 1.0e-10

defn safe_log(x) do
  Nx.log(Nx.max(x, @epsilon))
end

defn safe_divide(a, b) do
  Nx.divide(a, Nx.max(Nx.abs(b), @epsilon))
end
```

**Usage**: Anywhere division or log might receive zero.

### Pattern 2: Log-Space Computation

```elixir
# Bad: underflow risk
defn kl_bad(p, q) do
  ratio = Nx.divide(p, q)
  Nx.sum(Nx.multiply(p, Nx.log(ratio)))
end

# Good: log-space
defn kl_good(log_p, log_q) do
  p = Nx.exp(log_p)
  log_ratio = Nx.subtract(log_p, log_q)
  Nx.sum(Nx.multiply(p, log_ratio))
end
```

**Usage**: KL divergence, entropy, cross-entropy.

### Pattern 3: Log-Sum-Exp Trick

```elixir
# Bad: overflow risk
defn softmax_bad(x) do
  exp_x = Nx.exp(x)
  Nx.divide(exp_x, Nx.sum(exp_x))
end

# Good: subtract max
defn softmax_good(x) do
  max_x = Nx.reduce_max(x, axes: [-1], keep_axes: true)
  exp_x = Nx.exp(Nx.subtract(x, max_x))
  Nx.divide(exp_x, Nx.sum(exp_x, axes: [-1], keep_axes: true))
end

# Best: use Nx.logsumexp
defn log_softmax_best(x) do
  Nx.subtract(x, Nx.logsumexp(x, axes: [-1], keep_axes: true))
end
```

**Usage**: Any softmax or normalization operation.

### Pattern 4: Gradient Clipping at Source

```elixir
# Prevent exploding gradients from extreme values
defn stable_l2(tensor, opts) do
  clip = opts[:clip] || 1.0e6
  clipped = Nx.clip(tensor, -clip, clip)
  Nx.sum(Nx.power(clipped, 2))
end
```

**Usage**: L2 penalty, squared operations.

### Pattern 5: Masking Invalid Values

```elixir
# Handle 0 * log(0) = NaN case
defn entropy_stable(logprobs) do
  p = Nx.exp(logprobs)

  # Where p ≈ 0, contribution should be 0
  # Mask out very negative logprobs
  valid_mask = Nx.greater(logprobs, -100)
  safe_logprobs = Nx.select(valid_mask, logprobs, Nx.tensor(0.0))

  pointwise = Nx.negate(Nx.multiply(p, safe_logprobs))
  Nx.sum(pointwise, axes: [-1])
end
```

**Usage**: Entropy, cross-entropy where 0*log(0) appears.

---

## Function-Specific Stability

### L1 Penalty

**Concern**: Gradient discontinuity at zero.

```elixir
# Nx.sign(0) = 0, which is the correct subgradient
# No special handling needed
defn l1(tensor, opts) do
  Nx.sum(Nx.abs(tensor))  # Stable
end
```

**Edge Cases**:
- Input is all zeros: Returns 0 (correct)
- Input contains NaN: Propagates NaN
- Input contains Inf: Returns Inf

### L2 Penalty

**Concern**: Squaring large values causes overflow.

```elixir
defn l2_stable(tensor, opts) do
  clip = opts[:clip] || nil

  tensor = if clip do
    Nx.clip(tensor, -clip, clip)
  else
    tensor
  end

  Nx.sum(Nx.power(tensor, 2))
end
```

**Overflow Thresholds**:
| Type | Max safe value | Squared |
|------|----------------|---------|
| f16 | ~256 | ~65536 |
| f32 | ~1.8e19 | ~3.4e38 |
| bf16 | ~256 | ~65536 |

**Recommendation**: Default clip to 1e6 for f32.

### KL Divergence

**Concerns**:
1. Q = 0 where P > 0 → Inf
2. P = 0 → 0 * -Inf = NaN
3. Log of small values → -Inf

```elixir
defn kl_stable(p_log, q_log, opts) do
  epsilon = opts[:epsilon] || 1.0e-10

  p = Nx.exp(p_log)

  # Safe log ratio
  log_ratio = Nx.subtract(p_log, q_log)

  # Clamp extreme ratios
  log_ratio_safe = Nx.clip(log_ratio, -100, 100)

  # Mask zero probabilities
  valid = Nx.greater(p_log, -50)  # p > ~1e-22
  contribution = Nx.multiply(p, log_ratio_safe)
  contribution = Nx.select(valid, contribution, Nx.tensor(0.0))

  Nx.sum(contribution, axes: [-1])
end
```

### Entropy

**Concerns**:
1. p = 0: 0 * log(0) = 0 by convention
2. p = 1: log(1) = 0, contribution = 0

```elixir
defn entropy_stable(logprobs, opts) do
  p = Nx.exp(logprobs)

  # 0 * -inf handling via masking
  safe_threshold = -100
  valid = Nx.greater(logprobs, safe_threshold)

  # Where valid: -p * log_p
  # Where invalid: 0
  pointwise = Nx.negate(Nx.multiply(p, logprobs))
  masked = Nx.select(valid, pointwise, Nx.tensor(0.0))

  Nx.sum(masked, axes: [-1])
end
```

### Orthogonality

**Concerns**:
1. Zero vectors → NaN after normalization
2. Very large Gram matrix values

```elixir
defn orthogonality_stable(tensor, opts) do
  epsilon = 1.0e-8

  # Safe normalization
  norms = Nx.sqrt(Nx.sum(Nx.power(tensor, 2), axes: [1], keep_axes: true))
  safe_norms = Nx.max(norms, epsilon)
  normalized = Nx.divide(tensor, safe_norms)

  # Gram matrix
  gram = Nx.dot(normalized, [1], normalized, [1])

  # Clamp to [-1, 1] range (numerical errors can exceed)
  gram = Nx.clip(gram, -1.0, 1.0)

  # Off-diagonal penalty
  n = Nx.axis_size(gram, 0)
  identity = Nx.eye(n)
  off_diag = Nx.subtract(gram, Nx.multiply(gram, identity))

  Nx.sum(Nx.power(off_diag, 2))
end
```

---

## Gradient Stability

### Gradient of L1

```elixir
# ∂|x|/∂x = sign(x)
# At x=0, subgradient is any value in [-1, 1]
# Nx.sign(0) = 0, which works well in practice
```

**Issue**: Near zero, gradients can oscillate. Consider smooth approximation:

```elixir
# Huber-style smooth L1
defn smooth_l1(tensor, opts) do
  delta = opts[:delta] || 1.0

  abs_x = Nx.abs(tensor)
  quadratic = Nx.divide(Nx.power(tensor, 2), 2 * delta)
  linear = Nx.subtract(abs_x, delta / 2)

  Nx.sum(Nx.select(Nx.less(abs_x, delta), quadratic, linear))
end
```

### Gradient of Log

```elixir
# ∂log(x)/∂x = 1/x
# As x → 0, gradient → Inf
```

**Solution**: Always work in log-space for distributions:

```elixir
# Instead of computing log(softmax(x)), use:
log_probs = Nx.subtract(x, Nx.logsumexp(x, axes: [-1], keep_axes: true))
```

### Gradient of Division

```elixir
# ∂(a/b)/∂b = -a/b²
# As b → 0, gradient → Inf
```

**Solution**: Add epsilon to denominator:

```elixir
defn safe_ratio(a, b) do
  Nx.divide(a, Nx.add(b, 1.0e-10))
end
```

---

## Testing Stability

### Edge Case Test Matrix

| Input Type | Example | Expected Behavior |
|------------|---------|-------------------|
| All zeros | `[0, 0, 0]` | Returns 0 or valid penalty |
| All ones | `[1, 1, 1]` | Returns valid penalty |
| Single element | `[5.0]` | Works correctly |
| Very small | `[1e-30, 1e-30]` | No underflow/NaN |
| Very large | `[1e30, 1e30]` | No overflow/Inf |
| Mixed signs | `[-1, 0, 1]` | Correct handling |
| Contains NaN | `[1, NaN, 2]` | Propagates or handles |
| Contains Inf | `[1, Inf, 2]` | Propagates or handles |
| Empty tensor | `[]` | Returns 0 or error |

### Stability Test Template

```elixir
describe "stability" do
  test "handles zero input" do
    result = penalty_fn.(Nx.tensor([0.0, 0.0, 0.0]))
    assert_finite(result)
  end

  test "handles very small values" do
    result = penalty_fn.(Nx.tensor([1.0e-30, 1.0e-30]))
    assert_finite(result)
  end

  test "handles very large values" do
    result = penalty_fn.(Nx.tensor([1.0e30, 1.0e30]))
    assert_finite(result)
  end

  test "gradient is finite at zero" do
    grad_fn = Nx.Defn.grad(fn x -> penalty_fn.(x) end)
    grads = grad_fn.(Nx.tensor([0.0, 0.0, 0.0]))
    assert_finite(grads)
  end

  test "gradient is finite for small values" do
    grad_fn = Nx.Defn.grad(fn x -> penalty_fn.(x) end)
    grads = grad_fn.(Nx.tensor([1.0e-10, 1.0e-10]))
    assert_finite(grads)
  end
end
```

---

## Configuration Options

### Stability-Related Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `:epsilon` | float | `1.0e-10` | Small value for division/log |
| `:clip` | float | `nil` | Max value before operations |
| `:nan_policy` | atom | `:propagate` | `:propagate`, `:zero`, `:raise` |
| `:inf_policy` | atom | `:propagate` | `:propagate`, `:clip`, `:raise` |

### Implementation

```elixir
defn apply_nan_policy(tensor, policy) do
  case policy do
    :propagate -> tensor
    :zero -> Nx.select(Nx.is_nan(tensor), Nx.tensor(0.0), tensor)
    # :raise handled at validation layer
  end
end

defn apply_inf_policy(tensor, policy, clip_value \\ 1.0e10) do
  case policy do
    :propagate -> tensor
    :clip -> Nx.clip(tensor, -clip_value, clip_value)
    # :raise handled at validation layer
  end
end
```

---

## Type-Specific Considerations

### Float16 / BFloat16

**Reduced precision** limits safe value ranges:

| Type | Mantissa | Safe range |
|------|----------|------------|
| f16 | 10 bits | ±65504 |
| bf16 | 7 bits | ±3.4e38 (but less precise) |

**Recommendations**:
1. Cast intermediate computations to f32
2. Use aggressive clipping
3. Test specifically on f16/bf16 backends

```elixir
defn l2_mixed_precision(tensor, opts) do
  # Upcast for computation
  tensor_f32 = Nx.as_type(tensor, :f32)
  result = Nx.sum(Nx.power(tensor_f32, 2))
  # Downcast result
  Nx.as_type(result, Nx.type(tensor))
end
```

---

## Validation Layer

### Input Validation (outside defn)

```elixir
defmodule NxPenalties.Validation do
  def validate_tensor!(tensor, name \\ "tensor") do
    unless is_struct(tensor, Nx.Tensor) do
      raise ArgumentError, "#{name} must be an Nx.Tensor"
    end

    if has_nan?(tensor) do
      raise ArgumentError, "#{name} contains NaN values"
    end

    if has_inf?(tensor) do
      raise ArgumentError, "#{name} contains Inf values"
    end

    tensor
  end

  def has_nan?(tensor) do
    tensor |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 1
  end

  def has_inf?(tensor) do
    tensor |> Nx.is_infinity() |> Nx.any() |> Nx.to_number() == 1
  end
end
```

---

## Checklist

- [ ] All functions handle zero input gracefully
- [ ] All functions handle extreme values (1e-30, 1e30)
- [ ] Log operations use log-space or epsilon clamping
- [ ] Division operations add epsilon to denominator
- [ ] L2 operations optionally clip large values
- [ ] Entropy/KL handle 0*log(0) case
- [ ] Gradients are finite for all valid inputs
- [ ] Edge case tests cover all scenarios
- [ ] Type-specific tests for f16/bf16
- [ ] Documentation specifies numerical behavior
