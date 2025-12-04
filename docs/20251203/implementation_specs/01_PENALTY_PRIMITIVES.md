# 01: Penalty Primitives Implementation Specification

## Overview

This document specifies the implementation of core penalty functions: L1, L2, and Elastic Net. These are the foundation of the library and must be:

1. **Numerically correct** - Match reference implementations
2. **JIT-compatible** - Pure `Nx.Defn` functions
3. **Configurable** - Support common use cases via options
4. **Efficient** - Minimize intermediate tensor allocations

## Module: NxPenalties.Penalties

### File Location
```
lib/nx_penalties/penalties.ex
```

### Module Structure
```elixir
defmodule NxPenalties.Penalties do
  @moduledoc """
  Core penalty functions for regularization.

  All functions operate on Nx tensors and return scalar penalty values.
  Designed for use inside `Nx.Defn` compiled functions.

  ## Numerical Stability

  These functions include safeguards against common numerical issues:
  - L1: Handles zero values correctly (subgradient = 0)
  - L2: Clips very large values before squaring to prevent overflow
  - Elastic Net: Inherits stability from L1 and L2

  ## Example

      import Nx.Defn

      defn regularized_loss(y_true, y_pred, params) do
        base_loss = Nx.mean(Nx.power(Nx.subtract(y_true, y_pred), 2))
        l2_penalty = NxPenalties.Penalties.l2(params, lambda: 0.01)
        Nx.add(base_loss, l2_penalty)
      end
  """

  import Nx.Defn
end
```

---

## Function: l1/2

### Specification

```elixir
@doc """
L1 penalty (Lasso regularization).

Computes λ * Σ|x| where λ is the regularization strength.

Encourages sparsity by driving small values to exactly zero.
The gradient is the sign function: ∂L1/∂x = λ * sign(x).

## Options

  * `:lambda` - Regularization strength. Default: `1.0`
    > **Note:** Primitives default to `lambda: 1.0` (unscaled). Use pipeline `weight`
    > as the primary scaling knob. Only set `lambda` if you need intrinsic scaling
    > within the penalty function itself.
  * `:reduction` - How to aggregate values. Default: `:sum`
    * `:sum` - Sum of absolute values
    * `:mean` - Mean of absolute values

## Examples

    iex> tensor = Nx.tensor([1.0, -2.0, 0.5, -0.5])
    iex> NxPenalties.Penalties.l1(tensor, lambda: 0.1)
    #Nx.Tensor<
      f32
      0.4
    >

## Gradient Note

At x=0, the subgradient is 0. Nx handles this correctly via `Nx.sign/1`.
"""
@spec l1(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
defn l1(tensor, opts \\ []) do
  lambda = opts[:lambda] || 1.0
  reduction = opts[:reduction] || :sum

  abs_values = Nx.abs(tensor)

  reduced = case reduction do
    :sum -> Nx.sum(abs_values)
    :mean -> Nx.mean(abs_values)
  end

  Nx.multiply(reduced, lambda)
end
```

### Implementation Notes

1. **Branch in Defn**: The `case` on `:reduction` will be resolved at compile time when `opts` is a compile-time constant. If `opts` is runtime, use `Nx.select/3`:
   ```elixir
   is_sum = opts[:reduction] == :sum
   Nx.select(is_sum, Nx.sum(abs_values), Nx.mean(abs_values))
   ```

2. **Type Preservation**: Result inherits type from input tensor. No explicit casting needed.

3. **Empty Tensor**: If input is empty, `Nx.sum` returns 0.0, which is correct.

### Test Cases

```elixir
describe "l1/2" do
  test "computes correct L1 norm with default lambda" do
    tensor = Nx.tensor([1.0, -2.0, 3.0])
    result = NxPenalties.Penalties.l1(tensor)
    # Expected: 1.0 * (1 + 2 + 3) = 6.0 (unscaled by default)
    assert_close(result, Nx.tensor(6.0))
  end

  test "respects custom lambda" do
    tensor = Nx.tensor([1.0, -1.0])
    result = NxPenalties.Penalties.l1(tensor, lambda: 0.5)
    # Expected: 0.5 * 2 = 1.0
    assert_close(result, Nx.tensor(1.0))
  end

  test "handles zero values" do
    tensor = Nx.tensor([0.0, 0.0, 1.0])
    result = NxPenalties.Penalties.l1(tensor, lambda: 1.0)
    assert_close(result, Nx.tensor(1.0))
  end

  test "mean reduction" do
    tensor = Nx.tensor([1.0, 2.0, 3.0])
    result = NxPenalties.Penalties.l1(tensor, lambda: 1.0, reduction: :mean)
    assert_close(result, Nx.tensor(2.0))
  end

  test "works with multidimensional tensors" do
    tensor = Nx.tensor([[1.0, -2.0], [-3.0, 4.0]])
    result = NxPenalties.Penalties.l1(tensor, lambda: 0.1)
    # Expected: 0.1 * 10 = 1.0
    assert_close(result, Nx.tensor(1.0))
  end

  test "gradient is sign function" do
    grad_fn = Nx.Defn.grad(fn x -> NxPenalties.Penalties.l1(x, lambda: 1.0) end)
    tensor = Nx.tensor([2.0, -3.0, 0.0])
    grads = grad_fn.(tensor)
    # Expected: [1.0, -1.0, 0.0]
    assert_close(grads, Nx.tensor([1.0, -1.0, 0.0]))
  end

  test "JIT compiles successfully" do
    jit_l1 = Nx.Defn.jit(&NxPenalties.Penalties.l1/2)
    tensor = Nx.tensor([1.0, 2.0])
    assert jit_l1.(tensor, lambda: 0.1)
  end
end
```

---

## Function: l2/2

### Specification

```elixir
@doc """
L2 penalty (Ridge/Tikhonov regularization).

Computes λ * Σx² where λ is the regularization strength.

Encourages small values without inducing sparsity. All values shrink
proportionally rather than being driven to zero.

## Options

  * `:lambda` - Regularization strength. Default: `1.0`
    > **Note:** Primitives default to `lambda: 1.0` (unscaled). Use pipeline `weight`
    > as the primary scaling knob.
  * `:reduction` - How to aggregate values. Default: `:sum`
    * `:sum` - Sum of squared values
    * `:mean` - Mean of squared values
  * `:clip` - Maximum absolute value before squaring. Default: `nil` (no clip)
    Useful for preventing overflow with very large values.

## Examples

    iex> tensor = Nx.tensor([1.0, 2.0, 3.0])
    iex> NxPenalties.Penalties.l2(tensor, lambda: 0.1)
    #Nx.Tensor<
      f32
      1.4
    >

## Gradient

The gradient is linear: ∂L2/∂x = 2λx
"""
@spec l2(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
defn l2(tensor, opts \\ []) do
  lambda = opts[:lambda] || 1.0
  reduction = opts[:reduction] || :sum
  clip_val = opts[:clip]

  # Optional clipping for numerical stability
  clipped = if clip_val do
    Nx.clip(tensor, -clip_val, clip_val)
  else
    tensor
  end

  squared = Nx.power(clipped, 2)

  reduced = case reduction do
    :sum -> Nx.sum(squared)
    :mean -> Nx.mean(squared)
  end

  Nx.multiply(reduced, lambda)
end
```

### Implementation Notes

1. **Overflow Prevention**: For fp16 tensors, squaring values > ~250 causes overflow. The `:clip` option addresses this.

2. **Relation to Weight Decay**: In AdamW, weight decay is applied differently (to params directly, not gradients). This L2 penalty is the classical loss-based formulation.

3. **Half-lambda Convention**: Some implementations use `0.5 * λ * Σx²` so that the gradient is `λx` instead of `2λx`. We use the non-halved version to match PyTorch's default. Document this clearly.

### Test Cases

```elixir
describe "l2/2" do
  test "computes correct L2 norm" do
    tensor = Nx.tensor([1.0, 2.0, 3.0])
    result = NxPenalties.Penalties.l2(tensor, lambda: 0.1)
    # Expected: 0.1 * (1 + 4 + 9) = 1.4
    assert_close(result, Nx.tensor(1.4))
  end

  test "handles negative values (squared)" do
    tensor = Nx.tensor([-2.0, -3.0])
    result = NxPenalties.Penalties.l2(tensor, lambda: 0.1)
    # Expected: 0.1 * (4 + 9) = 1.3
    assert_close(result, Nx.tensor(1.3))
  end

  test "clipping prevents overflow" do
    tensor = Nx.tensor([1000.0, 2000.0], type: :f16)
    # Without clip, this would overflow
    result = NxPenalties.Penalties.l2(tensor, lambda: 0.01, clip: 100.0)
    # Clipped to [100, 100], squared = [10000, 10000], sum * 0.01 = 200
    assert_close(result, Nx.tensor(200.0))
  end

  test "gradient is 2*lambda*x" do
    grad_fn = Nx.Defn.grad(fn x -> NxPenalties.Penalties.l2(x, lambda: 0.5) end)
    tensor = Nx.tensor([1.0, 2.0, 3.0])
    grads = grad_fn.(tensor)
    # Expected: 2 * 0.5 * [1, 2, 3] = [1, 2, 3]
    assert_close(grads, Nx.tensor([1.0, 2.0, 3.0]))
  end

  test "mean reduction" do
    tensor = Nx.tensor([1.0, 2.0])
    result = NxPenalties.Penalties.l2(tensor, lambda: 1.0, reduction: :mean)
    # Expected: mean([1, 4]) = 2.5
    assert_close(result, Nx.tensor(2.5))
  end
end
```

---

## Function: elastic_net/2

### Specification

```elixir
@doc """
Elastic Net penalty (combined L1 and L2).

Computes λ * (α * L1 + (1 - α) * L2) where:
- λ is the overall regularization strength
- α controls the L1/L2 balance (α=1 is pure L1, α=0 is pure L2)

Combines sparsity induction (L1) with smooth shrinkage (L2).

## Options

  * `:lambda` - Overall regularization strength. Default: `1.0`
    > **Note:** Primitives default to `lambda: 1.0` (unscaled). Use pipeline `weight`
    > as the primary scaling knob.
  * `:l1_ratio` - Balance between L1 and L2 (α). Default: `0.5`
    * `1.0` = pure L1
    * `0.5` = equal mix
    * `0.0` = pure L2
  * `:reduction` - How to aggregate values. Default: `:sum`

## Examples

    iex> tensor = Nx.tensor([1.0, -2.0, 3.0])
    iex> NxPenalties.Penalties.elastic_net(tensor, lambda: 0.1, l1_ratio: 0.5)
    # L1 component: 0.1 * 0.5 * 6 = 0.3
    # L2 component: 0.1 * 0.5 * 14 = 0.7
    # Total: 1.0
"""
@spec elastic_net(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
defn elastic_net(tensor, opts \\ []) do
  lambda = opts[:lambda] || 1.0
  l1_ratio = opts[:l1_ratio] || 0.5
  reduction = opts[:reduction] || :sum

  # Compute L1 component (without lambda, we'll apply combined lambda)
  abs_values = Nx.abs(tensor)
  l1_raw = case reduction do
    :sum -> Nx.sum(abs_values)
    :mean -> Nx.mean(abs_values)
  end

  # Compute L2 component
  squared = Nx.power(tensor, 2)
  l2_raw = case reduction do
    :sum -> Nx.sum(squared)
    :mean -> Nx.mean(squared)
  end

  # Combine: λ * (α * L1 + (1-α) * L2)
  l1_weighted = Nx.multiply(l1_raw, l1_ratio)
  l2_weighted = Nx.multiply(l2_raw, 1.0 - l1_ratio)
  combined = Nx.add(l1_weighted, l2_weighted)

  Nx.multiply(combined, lambda)
end
```

### Implementation Notes

1. **Efficiency**: Could call `l1/2` and `l2/2` internally, but inlining avoids duplicate tensor traversal.

2. **Gradient**: Gradient is `λ * (α * sign(x) + 2(1-α) * x)`. The L2 component ensures gradient is never zero except at origin.

3. **Use Case Guidance**:
   - `l1_ratio: 0.9-1.0` - When sparsity is primary goal
   - `l1_ratio: 0.5` - Balanced (good default)
   - `l1_ratio: 0.1-0.3` - Smoothness with slight sparsity

### Test Cases

```elixir
describe "elastic_net/2" do
  test "with l1_ratio=1.0 equals l1" do
    tensor = Nx.tensor([1.0, -2.0, 3.0])
    elastic = NxPenalties.Penalties.elastic_net(tensor, lambda: 0.1, l1_ratio: 1.0)
    l1 = NxPenalties.Penalties.l1(tensor, lambda: 0.1)
    assert_close(elastic, l1)
  end

  test "with l1_ratio=0.0 equals l2" do
    tensor = Nx.tensor([1.0, -2.0, 3.0])
    elastic = NxPenalties.Penalties.elastic_net(tensor, lambda: 0.1, l1_ratio: 0.0)
    l2 = NxPenalties.Penalties.l2(tensor, lambda: 0.1)
    assert_close(elastic, l2)
  end

  test "balanced ratio combines both" do
    tensor = Nx.tensor([1.0, 2.0])
    result = NxPenalties.Penalties.elastic_net(tensor, lambda: 1.0, l1_ratio: 0.5)
    # L1: 3, L2: 5, combined: 0.5*3 + 0.5*5 = 4.0
    assert_close(result, Nx.tensor(4.0))
  end

  test "gradient combines L1 and L2 gradients" do
    grad_fn = Nx.Defn.grad(fn x ->
      NxPenalties.Penalties.elastic_net(x, lambda: 1.0, l1_ratio: 0.5)
    end)
    tensor = Nx.tensor([2.0, -3.0])
    grads = grad_fn.(tensor)
    # L1 grad: [1, -1] * 0.5 = [0.5, -0.5]
    # L2 grad: [4, -6] * 0.5 = [2, -3]
    # Combined: [2.5, -3.5]
    assert_close(grads, Nx.tensor([2.5, -3.5]))
  end
end
```

---

## Validation Module

### File Location
```
lib/nx_penalties/penalties/validation.ex
```

### Purpose
Validate options outside of defn, before entering hot path.

```elixir
defmodule NxPenalties.Penalties.Validation do
  @moduledoc false

  @l1_schema [
    lambda: [type: :float, default: 1.0],
    reduction: [type: {:in, [:sum, :mean]}, default: :sum]
  ]

  @l2_schema [
    lambda: [type: :float, default: 1.0],
    reduction: [type: {:in, [:sum, :mean]}, default: :sum],
    clip: [type: {:or, [:float, nil]}, default: nil]
  ]

  @elastic_net_schema [
    lambda: [type: :float, default: 1.0],
    l1_ratio: [type: :float, default: 0.5],
    reduction: [type: {:in, [:sum, :mean]}, default: :sum]
  ]

  def validate_l1!(opts), do: NimbleOptions.validate!(opts, @l1_schema)
  def validate_l2!(opts), do: NimbleOptions.validate!(opts, @l2_schema)
  def validate_elastic_net!(opts), do: NimbleOptions.validate!(opts, @elastic_net_schema)
end
```

---

## Performance Benchmarks

### Expected Performance

| Operation | Tensor Size | Expected Time (CPU) | Expected Time (GPU) |
|-----------|-------------|---------------------|---------------------|
| `l1/2` | 1M floats | < 1ms | < 0.1ms |
| `l2/2` | 1M floats | < 1ms | < 0.1ms |
| `elastic_net/2` | 1M floats | < 2ms | < 0.2ms |

### Benchmark Code

```elixir
defmodule NxPenalties.Penalties.Bench do
  def run do
    tensor = Nx.random_uniform({1_000_000})

    Benchee.run(%{
      "l1" => fn -> NxPenalties.Penalties.l1(tensor) end,
      "l2" => fn -> NxPenalties.Penalties.l2(tensor) end,
      "elastic_net" => fn -> NxPenalties.Penalties.elastic_net(tensor) end
    })
  end
end
```

---

## Integration Checklist

- [ ] All functions have `@doc` with examples
- [ ] All functions have `@spec` type specifications
- [ ] All functions are `defn` (JIT-compatible)
- [ ] Test coverage includes edge cases (empty, zero, negative, large)
- [ ] Gradient tests verify correct autodiff behavior
- [ ] JIT compilation test for each function
- [ ] Benchmark baselines established
