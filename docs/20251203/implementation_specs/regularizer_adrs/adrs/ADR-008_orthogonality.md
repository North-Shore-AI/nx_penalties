# ADR-008: Orthogonality Regularizer

## Status

Proposed

## Context

Orthogonality regularization encourages learned representations to be orthogonal (uncorrelated). This is particularly relevant for LoRA fine-tuning because:

1. **LoRA structure** - LoRA adds low-rank matrices A and B where the update is BA. Orthogonality in these matrices improves conditioning.

2. **Representation diversity** - orthogonal features capture distinct information, reducing redundancy.

3. **Training stability** - orthogonal weight matrices have condition number 1, optimal for gradient flow.

4. **Capacity utilization** - prevents multiple dimensions from learning the same feature.

Since Tinker doesn't expose LoRA weights directly, we apply orthogonality constraints to output representations (logprobs or attention patterns if available).

## Decision

Implement `Tinkex.Regularizer.Orthogonality` for encouraging diverse, uncorrelated outputs.

### Interface

```elixir
defmodule Tinkex.Regularizer.Orthogonality do
  @behaviour Tinkex.Regularizer

  @moduledoc """
  Orthogonality regularizer for encouraging diverse representations.

  Penalizes correlation between different dimensions/positions in the output.

  ## Modes

  - `:soft` - penalize off-diagonal elements of correlation matrix
  - `:hard` - penalize deviation from identity matrix
  - `:spectral` - penalize deviation from uniform singular values

  ## Application to Logprobs

  For logprob outputs of shape [batch, seq, vocab]:
  - Computes correlation across vocabulary dimension
  - Encourages different sequence positions to have uncorrelated predictions

  ## Example

      %RegularizerSpec{
        fn: &Orthogonality.compute/3,
        weight: 0.01,
        name: "orthogonality",
        opts: [mode: :soft]
      }
  """

  @impl true
  def compute(_data, logprobs, opts \\ []) do
    mode = Keyword.get(opts, :mode, :soft)
    axis = Keyword.get(opts, :axis, :sequence)

    # Reshape for correlation computation
    matrix = prepare_matrix(logprobs, axis)

    case mode do
      :soft -> soft_orthogonality(matrix)
      :hard -> hard_orthogonality(matrix)
      :spectral -> spectral_orthogonality(matrix)
    end
  end

  @impl true
  def name, do: "orthogonality"

  # Prepare matrix for orthogonality computation
  defp prepare_matrix(logprobs, axis) do
    case axis do
      :sequence ->
        # [batch, seq, vocab] -> [batch*vocab, seq] or [seq, vocab]
        # We want to measure if different sequence positions are orthogonal
        case Nx.shape(logprobs) do
          {batch, seq, vocab} ->
            # Reshape to [seq, batch*vocab] - each row is a sequence position
            logprobs
            |> Nx.transpose(axes: [1, 0, 2])  # [seq, batch, vocab]
            |> Nx.reshape({seq, batch * vocab})

          {seq, vocab} ->
            logprobs  # Already [seq, vocab]
        end

      :vocabulary ->
        # Measure if vocabulary dimensions are orthogonal
        case Nx.shape(logprobs) do
          {batch, seq, vocab} ->
            logprobs
            |> Nx.reshape({batch * seq, vocab})
            |> Nx.transpose()  # [vocab, batch*seq]

          {seq, vocab} ->
            Nx.transpose(logprobs)  # [vocab, seq]
        end
    end
  end

  # Soft orthogonality: minimize off-diagonal correlations
  defp soft_orthogonality(matrix) do
    # Normalize rows
    norms = Nx.sqrt(Nx.sum(Nx.power(matrix, 2), axes: [1], keep_axes: true))
    norms_safe = Nx.max(norms, 1.0e-8)
    normalized = Nx.divide(matrix, norms_safe)

    # Compute correlation matrix: M @ M^T
    correlation = Nx.dot(normalized, Nx.transpose(normalized))

    # Get dimensions
    n = Nx.axis_size(correlation, 0)

    # Create identity matrix
    identity = Nx.eye(n, type: Nx.type(correlation))

    # Off-diagonal elements: correlation - identity
    off_diagonal = Nx.subtract(correlation, identity)

    # Frobenius norm of off-diagonal
    penalty = Nx.sum(Nx.power(off_diagonal, 2))

    # Normalize by n² - n (number of off-diagonal elements)
    normalized_penalty = Nx.divide(penalty, n * n - n)

    {normalized_penalty, %{
      "off_diagonal_norm" => Nx.to_number(penalty),
      "off_diagonal_mean" => Nx.to_number(normalized_penalty),
      "max_correlation" => Nx.to_number(Nx.reduce_max(Nx.abs(off_diagonal)))
    }}
  end

  # Hard orthogonality: penalize deviation from I
  defp hard_orthogonality(matrix) do
    normalized = normalize_rows(matrix)

    # M @ M^T should equal I
    gram = Nx.dot(normalized, Nx.transpose(normalized))
    n = Nx.axis_size(gram, 0)
    identity = Nx.eye(n, type: Nx.type(gram))

    # ||M @ M^T - I||²_F
    deviation = Nx.subtract(gram, identity)
    penalty = Nx.sum(Nx.power(deviation, 2))

    {penalty, %{
      "gram_deviation" => Nx.to_number(penalty),
      "diagonal_mean" => Nx.to_number(Nx.mean(Nx.take_diagonal(gram)))
    }}
  end

  # Spectral orthogonality: encourage uniform singular values
  defp spectral_orthogonality(matrix) do
    # Compute singular values via SVD
    # Note: Nx.LinAlg.svd may not be available on all backends

    # Fallback: use eigenvalues of M^T @ M
    gram = Nx.dot(Nx.transpose(matrix), matrix)

    # Approximate singular values via power iteration or eigendecomposition
    # For now, use trace-based approximation
    trace = Nx.sum(Nx.take_diagonal(gram))
    frobenius_sq = Nx.sum(Nx.power(matrix, 2))

    # For orthogonal matrix: trace = frobenius = n
    n = Nx.axis_size(matrix, 0) |> Nx.tensor(type: Nx.type(trace))

    # Penalty: deviation from expected relationship
    trace_penalty = Nx.power(Nx.subtract(trace, n), 2)

    {trace_penalty, %{
      "trace" => Nx.to_number(trace),
      "frobenius_squared" => Nx.to_number(frobenius_sq),
      "expected_trace" => Nx.to_number(n)
    }}
  end

  defp normalize_rows(matrix) do
    norms = Nx.sqrt(Nx.sum(Nx.power(matrix, 2), axes: [1], keep_axes: true))
    norms_safe = Nx.max(norms, 1.0e-8)
    Nx.divide(matrix, norms_safe)
  end
end
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `:mode` | atom | `:soft` | Orthogonality measure (`:soft`, `:hard`, `:spectral`) |
| `:axis` | atom | `:sequence` | Which axis to orthogonalize (`:sequence`, `:vocabulary`) |

## Consequences

### Positive

- Encourages diverse, non-redundant representations
- Improves gradient flow through orthogonal structure
- Can prevent mode collapse in certain settings
- Interpretable metrics (correlation coefficients)

### Negative

- Requires computing Gram matrix (O(n²) in sequence length)
- May conflict with desired correlations in output
- Not directly applicable to LoRA weights (API limitation)
- Spectral mode requires SVD (expensive or unavailable)

### Neutral

- Multiple modes for different tradeoffs
- Axis selection allows flexibility

## Implementation Notes

### Computational Complexity

| Mode | Complexity | Notes |
|------|------------|-------|
| `:soft` | O(n² × d) | n = positions, d = vocab |
| `:hard` | O(n² × d) | Same as soft |
| `:spectral` | O(n³) | Full SVD, often approximated |

For typical LLM outputs (seq=2048, vocab=32k), soft mode computes a 2048×2048 correlation matrix - manageable but not free.

### Relationship to LoRA

LoRA updates have the form: W' = W + BA where B ∈ R^{d×r}, A ∈ R^{r×k}.

Ideally, we'd regularize:
- B^T B ≈ I (columns of B orthogonal)
- A A^T ≈ I (rows of A orthogonal)

Without weight access, we use output orthogonality as a proxy, which encourages the combined effect to produce diverse outputs.

### Mode Selection Guide

| Mode | Behavior | Use Case |
|------|----------|----------|
| `:soft` | Only penalizes correlations | Allow diagonal variation |
| `:hard` | Enforce M@M^T = I | Strict orthogonality |
| `:spectral` | Uniform singular values | Isotropic representations |

### Typical Weight Range

| Weight | Effect |
|--------|--------|
| 0.001 | Light diversity encouragement |
| 0.01 | Moderate orthogonality pressure |
| 0.1 | Strong, may affect task performance |

## Alternatives Considered

### 1. Covariance Regularization
Penalize covariance instead of correlation. Doesn't account for scale differences.

### 2. Mutual Information Minimization
Minimize I(X_i; X_j) between dimensions. Requires density estimation, complex.

### 3. Decorrelation via Batch Normalization
Whitening transformation in forward pass. Not applicable to Tinker's API.

### 4. SRIP (Spectral Restricted Isometry Property)
Advanced orthogonality for deep networks. Too complex for initial implementation.

## References

- Bansal et al. (2018). "Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?"
- Saxe et al. (2013). "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks"
- Arjovsky et al. (2016). "Unitary Evolution Recurrent Neural Networks"
- Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"
