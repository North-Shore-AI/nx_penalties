# 03: Constraints Implementation Specification

## Overview

This document specifies structural constraint penalties: orthogonality, gradient penalty, and consistency. These are more complex than basic penalties, requiring:

1. **Gram matrix computation** (orthogonality)
2. **Second-order gradients** (gradient penalty)
3. **Paired inputs** (consistency)

## Module: NxPenalties.Constraints

### File Location
```
lib/nx_penalties/constraints.ex
```

### Module Structure
```elixir
defmodule NxPenalties.Constraints do
  @moduledoc """
  Structural constraint penalties for representations.

  These penalties enforce properties on learned representations beyond
  simple magnitude constraints. They are typically more computationally
  expensive than basic L1/L2 penalties.

  ## Computational Complexity

  | Function | Complexity | Memory |
  |----------|------------|--------|
  | `orthogonality/2` | O(n²d) | O(n²) |
  | `gradient_penalty/3` | O(forward_pass) | 2x training |
  | `consistency/3` | O(d) | O(d) |

  Where n = sequence length, d = dimension.
  """

  import Nx.Defn
end
```

---

## Function: orthogonality/2

### Specification

```elixir
@doc """
Orthogonality penalty for encouraging uncorrelated representations.

Penalizes off-diagonal elements of the Gram matrix (normalized dot products).
Encourages different dimensions/positions to capture distinct information.

## Modes

  * `:soft` - Only penalize off-diagonal correlations (default)
  * `:hard` - Penalize deviation from identity matrix (includes diagonal)

## Options

  * `:mode` - `:soft` or `:hard`. Default: `:soft`
  * `:axis` - Which axis represents features. Default: `:last`
    * `:last` - Last axis is feature dimension
    * `integer` - Specific axis index
  * `:normalize` - Normalize rows before computing Gram. Default: `true`

## Examples

    # Encourage orthogonal token representations
    penalty = NxPenalties.Constraints.orthogonality(hidden_states)

## Mathematics

For matrix X with rows x_i:
- Gram matrix: G = X @ X^T (after row normalization)
- Soft penalty: ||G - I||²_F excluding diagonal = Σ_{i≠j} G_{ij}²
- Hard penalty: ||G - I||²_F = Σ_{i,j} (G_{ij} - δ_{ij})²
"""
@spec orthogonality(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
defn orthogonality(tensor, opts \\ []) do
  mode = opts[:mode] || :soft
  normalize_rows = opts[:normalize] || true

  # Reshape to 2D: [n_samples, n_features]
  original_shape = Nx.shape(tensor)
  rank = tuple_size(original_shape)

  matrix = case rank do
    2 -> tensor
    3 ->
      {batch, seq, dim} = original_shape
      Nx.reshape(tensor, {batch * seq, dim})
    _ ->
      # Flatten all but last dimension
      last_dim = elem(original_shape, rank - 1)
      other_dims = div(Nx.size(tensor), last_dim)
      Nx.reshape(tensor, {other_dims, last_dim})
  end

  # Optionally normalize rows
  matrix = if normalize_rows do
    norms = Nx.sqrt(Nx.sum(Nx.power(matrix, 2), axes: [1], keep_axes: true))
    safe_norms = Nx.max(norms, 1.0e-8)
    Nx.divide(matrix, safe_norms)
  else
    matrix
  end

  # Compute Gram matrix: G = M @ M^T
  gram = Nx.dot(matrix, [1], matrix, [1])
  n = Nx.axis_size(gram, 0)

  case mode do
    :soft ->
      # Penalize only off-diagonal
      identity = Nx.eye(n)
      off_diagonal = Nx.subtract(gram, Nx.multiply(gram, identity))
      Nx.sum(Nx.power(off_diagonal, 2))

    :hard ->
      # Penalize deviation from identity
      identity = Nx.eye(n)
      deviation = Nx.subtract(gram, identity)
      Nx.sum(Nx.power(deviation, 2))
  end
end
```

### Implementation Notes

1. **Reshape Strategy**: The function accepts arbitrary tensor ranks and flattens appropriately.

2. **Normalization**: Row normalization converts dot products to cosine similarities, making the penalty scale-invariant.

3. **Memory**: Gram matrix is O(n²) where n is the number of samples/positions. For long sequences, this can be large.

4. **Sparse Approximation**: For very large n, consider stochastic approximation (random subset of pairs).

### Test Cases

```elixir
describe "orthogonality/2" do
  test "identity-like input has low penalty" do
    # Rows already orthogonal
    tensor = Nx.eye(4)
    result = NxPenalties.Constraints.orthogonality(tensor, mode: :hard)
    assert_close(result, Nx.tensor(0.0), atol: 1.0e-5)
  end

  test "identical rows have high penalty" do
    # All rows the same = maximally correlated
    tensor = Nx.broadcast(Nx.tensor([1.0, 0.0, 0.0, 0.0]), {4, 4})
    result = NxPenalties.Constraints.orthogonality(tensor, mode: :soft)
    # After normalization, all rows identical, Gram is all 1s
    # Off-diagonal: 12 elements, each = 1, penalty = 12
    assert Nx.to_number(result) > 10.0
  end

  test "soft mode ignores diagonal" do
    # Gram = I means soft penalty = 0
    tensor = Nx.eye(3)
    soft = NxPenalties.Constraints.orthogonality(tensor, mode: :soft)
    assert_close(soft, Nx.tensor(0.0), atol: 1.0e-5)
  end

  test "handles 3D input" do
    tensor = Nx.random_uniform({2, 4, 8})  # batch, seq, dim
    result = NxPenalties.Constraints.orthogonality(tensor)
    assert Nx.shape(result) == {}  # scalar
  end

  test "gradient flows" do
    grad_fn = Nx.Defn.grad(fn x ->
      NxPenalties.Constraints.orthogonality(x)
    end)
    tensor = Nx.random_uniform({4, 8})
    grads = grad_fn.(tensor)
    assert Nx.shape(grads) == {4, 8}
  end
end
```

---

## Function: gradient_penalty/3

### Specification

```elixir
@doc """
Gradient penalty for Lipschitz smoothness (WGAN-GP style).

Penalizes large gradients to encourage smooth functions. The penalty
is computed as (||∇f|| - target)² where target is typically 1.

## Options

  * `:target_norm` - Target gradient norm. Default: `1.0`
  * `:mode` - How to compute gradient. Default: `:output`
    * `:output` - Gradient of sum of outputs
    * `:custom` - User provides gradient computation

## Notes

- **Expensive**: Requires computing gradients, effectively doubling backward pass
- **Memory**: Stores intermediate activations for gradient computation
- Use sparingly, perhaps every N training steps

## Example

    # In a training step
    defn training_step(params, x, y) do
      {y_pred, grad_norm_penalty} = with_gradient_penalty(params, x, fn params, x ->
        model_forward(params, x)
      end)
      # ...
    end
"""
@spec gradient_penalty(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
defn gradient_penalty(input, output, opts \\ []) do
  target_norm = opts[:target_norm] || 1.0

  # Compute gradient of output w.r.t. input
  # This requires the output to be a scalar or we sum it
  scalar_output = Nx.sum(output)

  # Use Nx.Defn.grad to get gradients
  # Note: This is tricky - we need the gradient of a function
  # In practice, this would be done differently in real usage

  # Compute L2 norm of output (proxy for gradient magnitude)
  # Real implementation would use actual gradients
  grad_proxy = output  # Placeholder - real impl needs grad computation

  grad_norm = Nx.sqrt(Nx.sum(Nx.power(grad_proxy, 2)))

  # Penalty: (||grad|| - target)²
  deviation = Nx.subtract(grad_norm, target_norm)
  Nx.power(deviation, 2)
end

@doc """
Helper to compute gradient penalty within a forward pass.

Wraps a function and returns both output and gradient penalty.
"""
def with_gradient_penalty(input, forward_fn, opts \\ []) do
  target_norm = opts[:target_norm] || 1.0

  # Compute forward pass
  output = forward_fn.(input)

  # Compute gradient of output w.r.t input
  grad_fn = Nx.Defn.grad(fn x ->
    forward_fn.(x) |> Nx.sum()
  end)
  gradients = grad_fn.(input)

  # Gradient norm
  grad_norm = Nx.sqrt(Nx.sum(Nx.power(gradients, 2)))

  # Penalty
  penalty = Nx.power(Nx.subtract(grad_norm, target_norm), 2)

  {output, penalty}
end
```

### Implementation Notes

1. **True Gradient Penalty**: The actual WGAN-GP penalty requires computing gradients of the model output w.r.t. inputs. This is a second-order gradient (gradient of gradient).

2. **Nx.Defn.grad Nesting**: Nx supports nested gradient computation, but it significantly increases memory and compute.

3. **Interpolation**: WGAN-GP computes gradients at interpolated points between real and fake samples. This requires access to both distributions.

4. **Practical Usage**: Due to complexity, provide a high-level `with_gradient_penalty/3` helper that wraps the forward function.

### Alternative: Simplified Gradient Magnitude

```elixir
@doc """
Simplified gradient magnitude penalty.

Instead of computing actual gradients, penalizes large output magnitudes
as a proxy for gradient magnitude. Much cheaper but less precise.
"""
defn output_magnitude_penalty(output, opts \\ []) do
  target = opts[:target] || 1.0
  magnitude = Nx.sqrt(Nx.mean(Nx.power(output, 2)))
  Nx.power(Nx.subtract(magnitude, target), 2)
end
```

---

## Function: consistency/3

### Specification

```elixir
@doc """
Consistency penalty for paired inputs.

Penalizes divergence between outputs of original and augmented inputs.
Encourages robust, stable predictions.

## Options

  * `:metric` - Distance metric. Default: `:mse`
    * `:mse` - Mean squared error
    * `:l1` - L1 distance
    * `:kl` - KL divergence (if outputs are logprobs)
    * `:cosine` - Cosine distance (1 - cosine similarity)
  * `:reduction` - Reduction method. Default: `:mean`

## Examples

    # Penalize difference between clean and noisy input predictions
    clean_output = model(clean_input)
    noisy_output = model(add_noise(clean_input))
    penalty = NxPenalties.Constraints.consistency(clean_output, noisy_output)

## Use Cases

  - Semi-supervised learning
  - Data augmentation consistency
  - Temporal consistency (video)
  - Adversarial robustness
"""
@spec consistency(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
defn consistency(output1, output2, opts \\ []) do
  metric = opts[:metric] || :mse
  reduction = opts[:reduction] || :mean

  divergence = case metric do
    :mse ->
      diff = Nx.subtract(output1, output2)
      Nx.power(diff, 2)

    :l1 ->
      Nx.abs(Nx.subtract(output1, output2))

    :kl ->
      # Assume outputs are logprobs
      p = Nx.exp(output1)
      log_ratio = Nx.subtract(output1, output2)
      Nx.multiply(p, log_ratio)

    :cosine ->
      # Flatten to vectors for cosine
      flat1 = Nx.reshape(output1, {:auto})
      flat2 = Nx.reshape(output2, {:auto})

      dot = Nx.sum(Nx.multiply(flat1, flat2))
      norm1 = Nx.sqrt(Nx.sum(Nx.power(flat1, 2)))
      norm2 = Nx.sqrt(Nx.sum(Nx.power(flat2, 2)))

      cosine_sim = Nx.divide(dot, Nx.multiply(Nx.max(norm1, 1.0e-8), Nx.max(norm2, 1.0e-8)))
      # Return as scalar, reduction handled below
      Nx.reshape(Nx.subtract(1.0, cosine_sim), {})
  end

  # Apply reduction
  case {metric, reduction} do
    {:cosine, _} ->
      # Cosine already returns scalar
      divergence

    {_, :none} ->
      divergence

    {_, :sum} ->
      Nx.sum(divergence)

    {_, :mean} ->
      Nx.mean(divergence)
  end
end
```

### Implementation Notes

1. **Paired Data**: User is responsible for providing paired outputs. Augmentation happens before calling this function.

2. **Symmetric vs Asymmetric**: MSE and L1 are symmetric. KL is not - consider which direction matters.

3. **Stop Gradient**: In semi-supervised settings, often one output has gradients stopped. This is caller's responsibility.

### Test Cases

```elixir
describe "consistency/3" do
  test "identical outputs have zero penalty" do
    output = Nx.random_uniform({4, 8})
    result = NxPenalties.Constraints.consistency(output, output)
    assert_close(result, Nx.tensor(0.0), atol: 1.0e-6)
  end

  test "MSE metric" do
    o1 = Nx.tensor([1.0, 2.0])
    o2 = Nx.tensor([2.0, 4.0])
    result = NxPenalties.Constraints.consistency(o1, o2, metric: :mse)
    # MSE: mean([1, 4]) = 2.5
    assert_close(result, Nx.tensor(2.5))
  end

  test "L1 metric" do
    o1 = Nx.tensor([1.0, 2.0])
    o2 = Nx.tensor([2.0, 4.0])
    result = NxPenalties.Constraints.consistency(o1, o2, metric: :l1)
    # L1: mean([1, 2]) = 1.5
    assert_close(result, Nx.tensor(1.5))
  end

  test "cosine metric" do
    o1 = Nx.tensor([1.0, 0.0])
    o2 = Nx.tensor([0.0, 1.0])
    result = NxPenalties.Constraints.consistency(o1, o2, metric: :cosine)
    # Orthogonal vectors: cosine = 0, distance = 1
    assert_close(result, Nx.tensor(1.0), atol: 1.0e-5)
  end

  test "same direction has zero cosine distance" do
    o1 = Nx.tensor([1.0, 2.0])
    o2 = Nx.tensor([2.0, 4.0])  # same direction, different magnitude
    result = NxPenalties.Constraints.consistency(o1, o2, metric: :cosine)
    assert_close(result, Nx.tensor(0.0), atol: 1.0e-5)
  end
end
```

---

## Performance Guidelines

### Orthogonality

| Sequence Length | Gram Matrix Size | Memory (f32) | Recommendation |
|-----------------|------------------|--------------|----------------|
| 128 | 128×128 | 64 KB | OK |
| 512 | 512×512 | 1 MB | OK |
| 2048 | 2048×2048 | 16 MB | Consider sampling |
| 8192 | 8192×8192 | 256 MB | Use stochastic approximation |

**Stochastic Approximation**:
```elixir
defn stochastic_orthogonality(tensor, opts \\ []) do
  sample_size = opts[:sample_size] || 256

  # Random sample of row indices
  n = Nx.axis_size(tensor, 0)
  indices = Nx.random_uniform({sample_size}, 0, n, type: :s32)
  sampled = Nx.take(tensor, indices, axis: 0)

  orthogonality(sampled, opts)
end
```

### Gradient Penalty

- Doubles backward pass memory
- Consider applying every N steps:
  ```elixir
  apply_gp? = rem(step, 10) == 0
  penalty = if apply_gp?, do: gradient_penalty(...), else: 0.0
  ```

### Consistency

- Linear in input size
- No special considerations
- Can batch multiple augmentation pairs

---

## Integration Checklist

- [ ] Orthogonality handles all common tensor ranks (2D, 3D, 4D)
- [ ] Gradient penalty has clear documentation on computational cost
- [ ] Consistency supports all common distance metrics
- [ ] Memory requirements documented for large inputs
- [ ] Stochastic approximation provided for orthogonality
- [ ] All functions have gradient tests
