defmodule NxPenalties do
  @moduledoc """
  Composable regularization penalties for the Nx ecosystem.

  NxPenalties provides a collection of penalty functions and a composition
  system for ML training regularization. It fills the gap between Axon
  (model definition) and Polaris (optimization) by providing loss-based
  regularization infrastructure.

  ## Quick Start

      # Individual penalties
      l1_penalty = NxPenalties.l1(tensor, lambda: 0.01)
      l2_penalty = NxPenalties.l2(tensor, lambda: 0.001)

      # Pipeline composition
      pipeline = NxPenalties.pipeline([
        {:l1, weight: 0.001},
        {:l2, weight: 0.01},
        {:entropy, weight: 0.1, opts: [mode: :penalty]}
      ])

      {total, metrics} = NxPenalties.compute(pipeline, tensor)

  ## Design Philosophy

  - **Primitives are unscaled**: All penalty functions default to `lambda: 1.0`
  - **Weight is the knob**: Use pipeline weights for scaling, not lambda
  - **Log-space inputs**: Divergence functions expect log-probabilities
  - **JIT compatible**: All core functions work with `Nx.Defn.jit/2`

  ## Available Penalties

  | Function | Purpose |
  |----------|---------|
  | `l1/2` | Lasso regularization (sparsity) |
  | `l2/2` | Ridge regularization (weight decay) |
  | `elastic_net/2` | Combined L1 + L2 |
  | `kl_divergence/3` | KL(P || Q) for distributions |
  | `js_divergence/3` | Symmetric Jensen-Shannon |
  | `entropy/2` | Shannon entropy (bonus/penalty mode) |
  | `gradient_penalty/3` | Lipschitz smoothness (expensive) |
  | `output_magnitude_penalty/2` | Cheaper gradient penalty proxy |
  | `interpolated_gradient_penalty/4` | WGAN-GP style penalty |

  ## Constraints & Tracking

  - `orthogonality/2` - Decorrelation penalty
  - `consistency/3` - Paired output consistency
  - Gradient norm tracking in pipelines (`track_grad_norms: true`)
  """

  alias NxPenalties.{Divergences, GradientPenalty, Penalties, Pipeline}

  # ============================================================================
  # Penalty Functions (Validated Wrappers)
  # ============================================================================

  @doc """
  L1 penalty (Lasso regularization).

  See `NxPenalties.Penalties.l1/2` for full documentation.
  """
  @spec l1(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  defdelegate l1(tensor, opts \\ []), to: Penalties

  @doc """
  L2 penalty (Ridge regularization).

  See `NxPenalties.Penalties.l2/2` for full documentation.
  """
  @spec l2(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  defdelegate l2(tensor, opts \\ []), to: Penalties

  @doc """
  Elastic Net penalty (combined L1 + L2).

  See `NxPenalties.Penalties.elastic_net/2` for full documentation.
  """
  @spec elastic_net(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  defdelegate elastic_net(tensor, opts \\ []), to: Penalties

  @doc """
  Kullback-Leibler divergence.

  See `NxPenalties.Divergences.kl_divergence/3` for full documentation.
  """
  @spec kl_divergence(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  defdelegate kl_divergence(p_logprobs, q_logprobs, opts \\ []), to: Divergences

  @doc """
  Jensen-Shannon divergence.

  See `NxPenalties.Divergences.js_divergence/3` for full documentation.
  """
  @spec js_divergence(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  defdelegate js_divergence(p_logprobs, q_logprobs, opts \\ []), to: Divergences

  @doc """
  Shannon entropy.

  See `NxPenalties.Divergences.entropy/2` for full documentation.
  """
  @spec entropy(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  defdelegate entropy(logprobs, opts \\ []), to: Divergences

  @doc """
  Gradient penalty for Lipschitz smoothness.

  See `NxPenalties.GradientPenalty.gradient_penalty/3` for full documentation.
  """
  @spec gradient_penalty((Nx.Tensor.t() -> Nx.Tensor.t()), Nx.Tensor.t(), keyword()) ::
          Nx.Tensor.t()
  defdelegate gradient_penalty(loss_fn, tensor, opts \\ []), to: GradientPenalty

  @doc """
  Output magnitude penalty (cheaper proxy for gradient penalty).

  See `NxPenalties.GradientPenalty.output_magnitude_penalty/2` for full documentation.
  """
  @spec output_magnitude_penalty(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  defdelegate output_magnitude_penalty(output, opts \\ []), to: GradientPenalty

  @doc """
  Interpolated gradient penalty (WGAN-GP style).

  See `NxPenalties.GradientPenalty.interpolated_gradient_penalty/4` for full documentation.
  """
  @spec interpolated_gradient_penalty(
          (Nx.Tensor.t() -> Nx.Tensor.t()),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          keyword()
        ) :: Nx.Tensor.t()
  defdelegate interpolated_gradient_penalty(loss_fn, tensor, reference, opts \\ []),
    to: GradientPenalty

  # ============================================================================
  # Pipeline API
  # ============================================================================

  @doc """
  Create a penalty pipeline from a declarative specification.

  ## Parameters

    * `specs` - List of penalty specifications

  ## Specification Format

  Each spec is a tuple: `{name, opts}` or just `name`

    * `name` - One of: `:l1`, `:l2`, `:elastic_net`, `:kl`, `:js`, `:entropy`
    * `opts` - Options:
      * `:weight` - Scaling factor. Default: `1.0`
      * `:opts` - Options passed to the penalty function
      * `:enabled` - Whether to include. Default: `true`

  ## Examples

      pipeline = NxPenalties.pipeline([
        {:l1, weight: 0.001},
        {:l2, weight: 0.01},
        {:entropy, weight: 0.1, opts: [mode: :penalty]}
      ])

      # Simple form
      pipeline = NxPenalties.pipeline([:l1, :l2])
  """
  @spec pipeline([{atom(), keyword()} | atom()]) :: Pipeline.t()
  def pipeline(specs) when is_list(specs) do
    Enum.reduce(specs, Pipeline.new(), fn spec, acc ->
      {name, opts} = normalize_spec(spec)
      penalty_fn = penalty_fn_for(name)
      Pipeline.add(acc, name, penalty_fn, opts)
    end)
  end

  defp normalize_spec({name, opts}) when is_atom(name) and is_list(opts), do: {name, opts}
  defp normalize_spec(name) when is_atom(name), do: {name, []}

  defp penalty_fn_for(:l1), do: &Penalties.l1/2
  defp penalty_fn_for(:l2), do: &Penalties.l2/2
  defp penalty_fn_for(:elastic_net), do: &Penalties.elastic_net/2
  defp penalty_fn_for(:kl), do: &Divergences.kl_divergence/3
  defp penalty_fn_for(:js), do: &Divergences.js_divergence/3
  defp penalty_fn_for(:entropy), do: &Divergences.entropy/2

  defp penalty_fn_for(name) do
    raise ArgumentError,
          "Unknown penalty: #{inspect(name)}. " <>
            "Available: :l1, :l2, :elastic_net, :kl, :js, :entropy"
  end

  @doc """
  Compute penalties from a pipeline.

  Delegates to `NxPenalties.Pipeline.compute/3`.

  ## Examples

      {total, metrics} = NxPenalties.compute(pipeline, tensor)
      {total, metrics} = NxPenalties.compute(pipeline, tensor, track_grad_norms: true)
  """
  @spec compute(Pipeline.t(), Nx.Tensor.t(), keyword()) :: {Nx.Tensor.t(), map()}
  defdelegate compute(pipeline, tensor, opts \\ []), to: Pipeline

  @doc """
  Compute only the total penalty (gradient-compatible).

  Delegates to `NxPenalties.Pipeline.compute_total/3`.
  """
  @spec compute_total(Pipeline.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  defdelegate compute_total(pipeline, tensor, opts \\ []), to: Pipeline

  # ============================================================================
  # Validation
  # ============================================================================

  @doc """
  Validate that a tensor contains no NaN or Inf values.

  ## Examples

      iex> NxPenalties.validate(Nx.tensor([1.0, 2.0, 3.0]))
      {:ok, #Nx.Tensor<...>}

      iex> NxPenalties.validate(Nx.Constants.nan({:f, 32}))
      {:error, :nan}

      iex> NxPenalties.validate(Nx.Constants.infinity({:f, 32}))
      {:error, :inf}

  ## Returns

    * `{:ok, tensor}` - Tensor is finite (no NaN or Inf)
    * `{:error, :nan}` - Tensor contains NaN values
    * `{:error, :inf}` - Tensor contains Inf values

  ## Notes

  NaN is checked first, so if a tensor contains both NaN and Inf,
  `{:error, :nan}` is returned.
  """
  @spec validate(Nx.Tensor.t()) :: {:ok, Nx.Tensor.t()} | {:error, :nan | :inf}
  def validate(tensor) do
    cond do
      has_nan?(tensor) -> {:error, :nan}
      has_inf?(tensor) -> {:error, :inf}
      true -> {:ok, tensor}
    end
  end

  defp has_nan?(tensor) do
    tensor
    |> Nx.is_nan()
    |> Nx.any()
    |> Nx.to_number() == 1
  end

  defp has_inf?(tensor) do
    tensor
    |> Nx.is_infinity()
    |> Nx.any()
    |> Nx.to_number() == 1
  end
end
