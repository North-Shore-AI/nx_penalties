defmodule NxPenalties.Divergences do
  @moduledoc """
  Information-theoretic divergence and entropy functions.

  All functions operate on log-probabilities for numerical stability.
  This design choice avoids underflow when working with probabilities
  close to zero.

  ## Input Format

  Functions expect **log-probabilities** (not raw probabilities):
  - Valid inputs: outputs from `Nx.log(Nx.softmax(logits))`
  - Invalid inputs: raw probability tensors

  ## Numerical Stability

  These functions include stability measures:
  - KL: Clamps log ratios to avoid Inf
  - JS: Uses log-space mixture computation
  - Entropy: Masks zero-probability contributions
  """

  import Nx.Defn

  @doc """
  Kullback-Leibler divergence: KL(P || Q).

  Measures how distribution P diverges from distribution Q.
  Not symmetric: KL(P||Q) ≠ KL(Q||P).

  ## Options

    * `:reduction` - How to aggregate over batches. Default: `:mean`
      * `:mean` - Mean over batch dimension
      * `:sum` - Sum over batch dimension
      * `:none` - Return per-sample values

  ## Examples

      iex> p_logprobs = Nx.log(Nx.tensor([0.4, 0.3, 0.2, 0.1]))
      iex> q_logprobs = Nx.log(Nx.tensor([0.25, 0.25, 0.25, 0.25]))
      iex> NxPenalties.Divergences.kl_divergence(p_logprobs, q_logprobs)

  ## Mathematical Definition

      KL(P || Q) = Σ p(x) * log(p(x) / q(x))
                 = Σ p(x) * (log_p(x) - log_q(x))
  """
  @spec kl_divergence(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  deftransform kl_divergence(p_logprobs, q_logprobs, opts \\ []) do
    reduction = Keyword.get(opts, :reduction, :mean)

    case reduction do
      :mean -> kl_mean_impl(p_logprobs, q_logprobs)
      :sum -> kl_sum_impl(p_logprobs, q_logprobs)
      :none -> kl_none_impl(p_logprobs, q_logprobs)
    end
  end

  defnp kl_mean_impl(p_logprobs, q_logprobs) do
    kl_per_sample = kl_none_impl(p_logprobs, q_logprobs)
    Nx.mean(kl_per_sample)
  end

  defnp kl_sum_impl(p_logprobs, q_logprobs) do
    kl_per_sample = kl_none_impl(p_logprobs, q_logprobs)
    Nx.sum(kl_per_sample)
  end

  defnp kl_none_impl(p_logprobs, q_logprobs) do
    # P as probabilities
    p = Nx.exp(p_logprobs)

    # Log ratio: log(p/q) = log_p - log_q
    log_ratio = Nx.subtract(p_logprobs, q_logprobs)

    # Clamp extreme values for stability
    log_ratio_safe = Nx.clip(log_ratio, -100.0, 100.0)

    # KL = Σ p * log(p/q), summed over the last axis (classes)
    pointwise = Nx.multiply(p, log_ratio_safe)

    # Mask near-zero probabilities to avoid 0 * -inf = NaN
    # Where p is very small, contribution should be 0
    valid_mask = Nx.greater(p, 1.0e-10)
    masked = Nx.select(valid_mask, pointwise, Nx.tensor(0.0))

    Nx.sum(masked, axes: [-1])
  end

  @doc """
  Jensen-Shannon divergence: JS(P || Q).

  A symmetric, bounded divergence measure. Defined as:
      JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
  where M = 0.5 * (P + Q).

  Bounded: 0 ≤ JS ≤ log(2) ≈ 0.693

  ## Options

    * `:reduction` - How to aggregate over batches. Default: `:mean`

  ## Examples

      iex> p_logprobs = Nx.log(Nx.tensor([0.4, 0.3, 0.2, 0.1]))
      iex> q_logprobs = Nx.log(Nx.tensor([0.25, 0.25, 0.25, 0.25]))
      iex> NxPenalties.Divergences.js_divergence(p_logprobs, q_logprobs)
  """
  @spec js_divergence(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  deftransform js_divergence(p_logprobs, q_logprobs, opts \\ []) do
    reduction = Keyword.get(opts, :reduction, :mean)

    case reduction do
      :mean -> js_mean_impl(p_logprobs, q_logprobs)
      :sum -> js_sum_impl(p_logprobs, q_logprobs)
      :none -> js_none_impl(p_logprobs, q_logprobs)
    end
  end

  defnp js_mean_impl(p_logprobs, q_logprobs) do
    js_per_sample = js_none_impl(p_logprobs, q_logprobs)
    Nx.mean(js_per_sample)
  end

  defnp js_sum_impl(p_logprobs, q_logprobs) do
    js_per_sample = js_none_impl(p_logprobs, q_logprobs)
    Nx.sum(js_per_sample)
  end

  defnp js_none_impl(p_logprobs, q_logprobs) do
    # Compute mixture M = 0.5 * P + 0.5 * Q in log space
    # log(M) = log(0.5 * exp(log_p) + 0.5 * exp(log_q))
    # Use logsumexp trick: log(a + b) = log(a) + log(1 + b/a)
    p = Nx.exp(p_logprobs)
    q = Nx.exp(q_logprobs)
    m = Nx.divide(Nx.add(p, q), 2.0)
    m_logprobs = Nx.log(Nx.max(m, 1.0e-10))

    # KL(P || M)
    kl_p_m = kl_none_impl(p_logprobs, m_logprobs)

    # KL(Q || M)
    kl_q_m = kl_none_impl(q_logprobs, m_logprobs)

    # JS = 0.5 * (KL(P||M) + KL(Q||M))
    Nx.multiply(Nx.add(kl_p_m, kl_q_m), 0.5)
  end

  @doc """
  Shannon entropy of a probability distribution.

  Measures uncertainty/randomness in a distribution.
  Higher entropy = more uniform/uncertain.

  ## Options

    * `:mode` - Whether to use as penalty or bonus. Default: `:bonus`
      * `:bonus` - Returns H(P) (positive, encourages high entropy)
      * `:penalty` - Returns -H(P) (negative, penalizes high entropy)
    * `:reduction` - How to aggregate over batches. Default: `:mean`
    * `:normalize` - Normalize entropy by maximum possible value. Default: `false`
      * `false` - Return raw entropy
      * `true` - Divide by log(vocab_size) to get result in [0, 1]

  ## Examples

      iex> logprobs = Nx.log(Nx.tensor([0.25, 0.25, 0.25, 0.25]))
      iex> NxPenalties.Divergences.entropy(logprobs)
      # Returns log(4) ≈ 1.386 (maximum entropy for 4 classes)

      iex> logprobs = Nx.log(Nx.tensor([0.25, 0.25, 0.25, 0.25]))
      iex> NxPenalties.Divergences.entropy(logprobs, normalize: true)
      # Returns 1.0 (normalized maximum entropy)

  ## Mathematical Definition

      H(P) = -Σ p(x) * log(p(x))

  In log space:
      H(P) = -Σ exp(log_p) * log_p

  When normalized:
      H_norm(P) = H(P) / log(vocab_size)
  """
  @spec entropy(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  deftransform entropy(logprobs, opts \\ []) do
    mode = Keyword.get(opts, :mode, :bonus)
    reduction = Keyword.get(opts, :reduction, :mean)
    normalize = Keyword.get(opts, :normalize, false)

    case {mode, reduction, normalize} do
      {:bonus, :mean, false} -> entropy_bonus_mean_impl(logprobs)
      {:bonus, :sum, false} -> entropy_bonus_sum_impl(logprobs)
      {:bonus, :none, false} -> entropy_bonus_none_impl(logprobs)
      {:penalty, :mean, false} -> entropy_penalty_mean_impl(logprobs)
      {:penalty, :sum, false} -> entropy_penalty_sum_impl(logprobs)
      {:penalty, :none, false} -> entropy_penalty_none_impl(logprobs)
      {:bonus, :mean, true} -> entropy_bonus_mean_normalized_impl(logprobs)
      {:bonus, :sum, true} -> entropy_bonus_sum_normalized_impl(logprobs)
      {:bonus, :none, true} -> entropy_bonus_none_normalized_impl(logprobs)
      {:penalty, :mean, true} -> entropy_penalty_mean_normalized_impl(logprobs)
      {:penalty, :sum, true} -> entropy_penalty_sum_normalized_impl(logprobs)
      {:penalty, :none, true} -> entropy_penalty_none_normalized_impl(logprobs)
    end
  end

  defnp entropy_bonus_mean_impl(logprobs) do
    Nx.mean(entropy_bonus_none_impl(logprobs))
  end

  defnp entropy_bonus_sum_impl(logprobs) do
    Nx.sum(entropy_bonus_none_impl(logprobs))
  end

  defnp entropy_bonus_none_impl(logprobs) do
    p = Nx.exp(logprobs)

    # H = -Σ p * log_p
    pointwise = Nx.multiply(Nx.negate(p), logprobs)

    # Handle 0 * -inf = NaN by masking
    valid_mask = Nx.greater(logprobs, -50.0)
    masked = Nx.select(valid_mask, pointwise, Nx.tensor(0.0))

    Nx.sum(masked, axes: [-1])
  end

  defnp entropy_penalty_mean_impl(logprobs) do
    Nx.negate(entropy_bonus_mean_impl(logprobs))
  end

  defnp entropy_penalty_sum_impl(logprobs) do
    Nx.negate(entropy_bonus_sum_impl(logprobs))
  end

  defnp entropy_penalty_none_impl(logprobs) do
    Nx.negate(entropy_bonus_none_impl(logprobs))
  end

  # Normalized implementations
  defnp entropy_bonus_mean_normalized_impl(logprobs) do
    Nx.mean(entropy_bonus_none_normalized_impl(logprobs))
  end

  defnp entropy_bonus_sum_normalized_impl(logprobs) do
    Nx.sum(entropy_bonus_none_normalized_impl(logprobs))
  end

  defnp entropy_bonus_none_normalized_impl(logprobs) do
    raw = entropy_bonus_none_impl(logprobs)
    vocab_size = Nx.axis_size(logprobs, -1)
    # Convert vocab_size to a scalar tensor then take log
    vocab_size_tensor = Nx.as_type(vocab_size, Nx.type(logprobs))
    max_entropy = Nx.log(vocab_size_tensor)
    Nx.divide(raw, max_entropy)
  end

  defnp entropy_penalty_mean_normalized_impl(logprobs) do
    Nx.negate(entropy_bonus_mean_normalized_impl(logprobs))
  end

  defnp entropy_penalty_sum_normalized_impl(logprobs) do
    Nx.negate(entropy_bonus_sum_normalized_impl(logprobs))
  end

  defnp entropy_penalty_none_normalized_impl(logprobs) do
    Nx.negate(entropy_bonus_none_normalized_impl(logprobs))
  end
end
