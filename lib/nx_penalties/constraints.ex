defmodule NxPenalties.Constraints do
  @moduledoc """
  Structural constraint penalties for representations.

  > **v0.2 Preview**: This module contains basic implementations.
  > Advanced options (stochastic approximation, custom metrics) coming in v0.2.

  These penalties enforce properties on learned representations beyond
  simple magnitude constraints. They are typically more computationally
  expensive than basic L1/L2 penalties.

  ## Available Constraints

  | Function | Purpose | Complexity |
  |----------|---------|------------|
  | `orthogonality/2` | Decorrelate representations | O(n²d) |
  | `consistency/3` | Paired output consistency | O(d) |

  ## Example

      # Encourage orthogonal token representations
      penalty = NxPenalties.Constraints.orthogonality(hidden_states)

      # Consistency between clean and augmented inputs
      penalty = NxPenalties.Constraints.consistency(clean_out, noisy_out)
  """

  import Nx.Defn

  @doc """
  Orthogonality penalty for encouraging uncorrelated representations.

  Penalizes off-diagonal elements of the Gram matrix (normalized dot products).
  Encourages different dimensions/positions to capture distinct information.

  ## Options

    * `:mode` - `:soft` or `:hard`. Default: `:soft`
      * `:soft` - Only penalize off-diagonal correlations
      * `:hard` - Penalize deviation from identity matrix
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
  deftransform orthogonality(tensor, opts \\ []) do
    mode = Keyword.get(opts, :mode, :soft)
    normalize = Keyword.get(opts, :normalize, true)

    case mode do
      :soft -> orthogonality_soft_impl(tensor, normalize)
      :hard -> orthogonality_hard_impl(tensor, normalize)
    end
  end

  # Helper to reshape tensor to 2D before numerical computation
  deftransform reshape_to_2d(tensor) do
    shape = Nx.shape(tensor)
    rank = tuple_size(shape)

    case rank do
      2 ->
        tensor

      3 ->
        {batch, seq, dim} = shape
        Nx.reshape(tensor, {batch * seq, dim})

      _ ->
        # Flatten all but last dimension
        last_dim = elem(shape, rank - 1)
        other_dims = div(Nx.size(tensor), last_dim)
        Nx.reshape(tensor, {other_dims, last_dim})
    end
  end

  deftransform orthogonality_soft_impl(tensor, normalize) do
    matrix = reshape_to_2d(tensor)
    matrix = if normalize, do: normalize_rows(matrix), else: matrix
    orthogonality_soft_compute(matrix)
  end

  defnp orthogonality_soft_compute(matrix) do
    # Compute Gram matrix: G = M @ M^T
    gram = Nx.dot(matrix, [1], matrix, [1])
    n = Nx.axis_size(gram, 0)

    # Penalize only off-diagonal
    identity = Nx.eye(n)
    off_diagonal = Nx.subtract(gram, Nx.multiply(gram, identity))
    Nx.sum(Nx.pow(off_diagonal, 2))
  end

  deftransform orthogonality_hard_impl(tensor, normalize) do
    matrix = reshape_to_2d(tensor)
    matrix = if normalize, do: normalize_rows(matrix), else: matrix
    orthogonality_hard_compute(matrix)
  end

  defnp orthogonality_hard_compute(matrix) do
    # Compute Gram matrix
    gram = Nx.dot(matrix, [1], matrix, [1])
    n = Nx.axis_size(gram, 0)

    # Penalize deviation from identity
    identity = Nx.eye(n)
    deviation = Nx.subtract(gram, identity)
    Nx.sum(Nx.pow(deviation, 2))
  end

  defnp normalize_rows(matrix) do
    norms = Nx.sqrt(Nx.sum(Nx.pow(matrix, 2), axes: [1], keep_axes: true))
    safe_norms = Nx.max(norms, 1.0e-8)
    Nx.divide(matrix, safe_norms)
  end

  @doc """
  Consistency penalty for paired inputs.

  Penalizes divergence between outputs of original and augmented inputs.
  Encourages robust, stable predictions.

  ## Options

    * `:metric` - Distance metric. Default: `:mse`
      * `:mse` - Mean squared error
      * `:l1` - L1 distance
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
  deftransform consistency(output1, output2, opts \\ []) do
    metric = Keyword.get(opts, :metric, :mse)
    reduction = Keyword.get(opts, :reduction, :mean)

    case {metric, reduction} do
      {:mse, :mean} -> consistency_mse_mean_impl(output1, output2)
      {:mse, :sum} -> consistency_mse_sum_impl(output1, output2)
      {:mse, :none} -> consistency_mse_none_impl(output1, output2)
      {:l1, :mean} -> consistency_l1_mean_impl(output1, output2)
      {:l1, :sum} -> consistency_l1_sum_impl(output1, output2)
      {:l1, :none} -> consistency_l1_none_impl(output1, output2)
      {:cosine, _} -> consistency_cosine_impl(output1, output2)
    end
  end

  defnp consistency_mse_mean_impl(o1, o2) do
    diff = Nx.subtract(o1, o2)
    Nx.mean(Nx.pow(diff, 2))
  end

  defnp consistency_mse_sum_impl(o1, o2) do
    diff = Nx.subtract(o1, o2)
    Nx.sum(Nx.pow(diff, 2))
  end

  defnp consistency_mse_none_impl(o1, o2) do
    diff = Nx.subtract(o1, o2)
    Nx.pow(diff, 2)
  end

  defnp consistency_l1_mean_impl(o1, o2) do
    Nx.mean(Nx.abs(Nx.subtract(o1, o2)))
  end

  defnp consistency_l1_sum_impl(o1, o2) do
    Nx.sum(Nx.abs(Nx.subtract(o1, o2)))
  end

  defnp consistency_l1_none_impl(o1, o2) do
    Nx.abs(Nx.subtract(o1, o2))
  end

  defnp consistency_cosine_impl(o1, o2) do
    flat1 = Nx.flatten(o1)
    flat2 = Nx.flatten(o2)

    dot = Nx.sum(Nx.multiply(flat1, flat2))
    norm1 = Nx.sqrt(Nx.sum(Nx.pow(flat1, 2)))
    norm2 = Nx.sqrt(Nx.sum(Nx.pow(flat2, 2)))

    cosine_sim = Nx.divide(dot, Nx.multiply(Nx.max(norm1, 1.0e-8), Nx.max(norm2, 1.0e-8)))
    Nx.subtract(1.0, cosine_sim)
  end
end
