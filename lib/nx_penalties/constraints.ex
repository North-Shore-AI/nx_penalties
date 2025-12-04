defmodule NxPenalties.Constraints do
  @moduledoc """
  Structural constraint penalties for representations.

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

    * `:mode` - `:soft`, `:hard`, or `:spectral`. Default: `:soft`
      * `:soft` - Only penalize off-diagonal correlations
      * `:hard` - Penalize deviation from identity matrix
      * `:spectral` - Encourages uniform singular values via trace analysis
    * `:normalize` - Normalize rows before computing Gram. Default: `true`
    * `:axis` - Which dimension to orthogonalize. Default: `:rows`
      * `:rows` - Orthogonalize across rows (default behavior)
      * `:sequence` - Orthogonalize across sequence positions (for 2D/3D tensors)
      * `:vocabulary` - Orthogonalize across vocabulary dimensions (for 2D/3D tensors)

  ## Examples

      # Default: orthogonalize rows
      penalty = NxPenalties.Constraints.orthogonality(hidden_states)

      # Orthogonalize sequence positions (each position uncorrelated)
      penalty = NxPenalties.Constraints.orthogonality(logits, axis: :sequence)

      # Orthogonalize vocabulary dimensions (each vocab dim uncorrelated)
      penalty = NxPenalties.Constraints.orthogonality(embeddings, axis: :vocabulary)

  ## Axis Details

  For 3D tensor `[batch, seq, vocab]`:

    * `:rows` - Flattens to 2D `[batch*seq, vocab]` then orthogonalizes
    * `:sequence` - Reshapes to `[seq, batch*vocab]` (orthogonalize positions)
    * `:vocabulary` - Reshapes to `[vocab, batch*seq]` (orthogonalize dimensions)

  For 2D tensor `[seq, vocab]`:

    * `:rows` - Uses tensor as-is
    * `:sequence` - Uses tensor as-is (rows are sequence positions)
    * `:vocabulary` - Transposes to `[vocab, seq]`

  ## Mathematics

  For matrix X with rows x_i:
  - Gram matrix: G = X @ X^T (after row normalization)
  - Soft penalty: ||G - I||²_F excluding diagonal = Σ_{i≠j} G_{ij}²
  - Hard penalty: ||G - I||²_F = Σ_{i,j} (G_{ij} - δ_{ij})²
  - Spectral penalty: (trace(G) - n)² + Σ_{i≠j} G_{ij}² (trace deviation + off-diagonal)
  """
  @spec orthogonality(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  deftransform orthogonality(tensor, opts \\ []) do
    mode = Keyword.get(opts, :mode, :soft)
    normalize = Keyword.get(opts, :normalize, true)
    axis = Keyword.get(opts, :axis, :rows)

    # Prepare matrix based on axis
    matrix = prepare_for_axis(tensor, axis)

    # Then apply mode (soft/hard/spectral)
    case mode do
      :soft -> orthogonality_soft_impl(matrix, normalize)
      :hard -> orthogonality_hard_impl(matrix, normalize)
      :spectral -> orthogonality_spectral_impl(matrix, normalize)
    end
  end

  # Prepare tensor for orthogonality computation based on axis
  deftransformp prepare_for_axis(tensor, axis) do
    shape = Nx.shape(tensor)
    rank = tuple_size(shape)

    case {axis, rank} do
      # Default: treat as matrix rows
      {:rows, 2} ->
        tensor

      {:rows, _} ->
        flatten_to_2d(tensor)

      # Sequence axis: orthogonalize sequence positions
      {:sequence, 3} ->
        {batch, seq, vocab} = shape

        tensor
        |> Nx.transpose(axes: [1, 0, 2])
        |> Nx.reshape({seq, batch * vocab})

      {:sequence, 2} ->
        tensor

      # Vocabulary axis: orthogonalize vocabulary dimensions  
      {:vocabulary, 3} ->
        {batch, seq, vocab} = shape

        tensor
        |> Nx.reshape({batch * seq, vocab})
        |> Nx.transpose()

      {:vocabulary, 2} ->
        Nx.transpose(tensor)

      # Fallback for other ranks
      {_, _} ->
        flatten_to_2d(tensor)
    end
  end

  deftransformp flatten_to_2d(tensor) do
    shape = Nx.shape(tensor)
    rank = tuple_size(shape)
    last_dim = elem(shape, rank - 1)
    other_dims = div(Nx.size(tensor), last_dim)
    Nx.reshape(tensor, {other_dims, last_dim})
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

  deftransform orthogonality_spectral_impl(tensor, normalize) do
    matrix = reshape_to_2d(tensor)
    matrix = if normalize, do: normalize_rows(matrix), else: matrix
    orthogonality_spectral_compute(matrix)
  end

  defnp orthogonality_spectral_compute(matrix) do
    # Compute Gram matrix: G = M @ M^T
    gram = Nx.dot(matrix, [1], matrix, [1])
    n = Nx.axis_size(gram, 0)

    # For orthogonal rows, trace of G should equal number of rows
    trace = Nx.sum(Nx.take_diagonal(gram))

    # Penalize deviation of trace from expected value n
    # Convert n to a tensor-compatible form
    n_scalar = Nx.as_type(n, Nx.type(trace))
    trace_deviation = Nx.pow(Nx.subtract(trace, n_scalar), 2)

    # Also penalize off-diagonal elements (correlation between rows)
    identity = Nx.eye(n, type: Nx.type(gram))
    off_diagonal = Nx.subtract(gram, Nx.multiply(gram, identity))
    off_diag_penalty = Nx.sum(Nx.pow(off_diagonal, 2))

    # Combine trace deviation and off-diagonal penalty
    # This approximates checking if singular values are all 1
    Nx.add(trace_deviation, off_diag_penalty)
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
      * `:kl` - Symmetric KL divergence (Jensen-Shannon style) for log-probabilities
    * `:reduction` - Reduction method. Default: `:mean`

  ## Examples

      # Penalize difference between clean and noisy input predictions
      clean_output = model(clean_input)
      noisy_output = model(add_noise(clean_input))
      penalty = NxPenalties.Constraints.consistency(clean_output, noisy_output)

      # KL divergence for probability distributions
      logprobs1 = model(input1)
      logprobs2 = model(input2)
      penalty = NxPenalties.Constraints.consistency(logprobs1, logprobs2, metric: :kl)

  ## Use Cases

    - Semi-supervised learning
    - Data augmentation consistency
    - Temporal consistency (video)
    - Adversarial robustness
    - Distribution matching (KL metric)
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
      {:kl, :mean} -> consistency_kl_mean_impl(output1, output2)
      {:kl, :sum} -> consistency_kl_sum_impl(output1, output2)
      {:kl, :none} -> consistency_kl_none_impl(output1, output2)
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

  defnp consistency_kl_none_impl(output1, output2) do
    # Treat outputs as log-probabilities
    p = Nx.exp(output1)
    q = Nx.exp(output2)

    # KL(P||Q) = Σ p * (log_p - log_q)
    kl_pq = Nx.sum(Nx.multiply(p, Nx.subtract(output1, output2)), axes: [-1])

    # KL(Q||P) = Σ q * (log_q - log_p)
    kl_qp = Nx.sum(Nx.multiply(q, Nx.subtract(output2, output1)), axes: [-1])

    # Symmetric: (KL(P||Q) + KL(Q||P)) / 2
    Nx.divide(Nx.add(kl_pq, kl_qp), 2.0)
  end

  defnp consistency_kl_mean_impl(output1, output2) do
    Nx.mean(consistency_kl_none_impl(output1, output2))
  end

  defnp consistency_kl_sum_impl(output1, output2) do
    Nx.sum(consistency_kl_none_impl(output1, output2))
  end
end
