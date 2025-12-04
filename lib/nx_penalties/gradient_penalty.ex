defmodule NxPenalties.GradientPenalty do
  @moduledoc """
  Gradient penalty regularizers for Lipschitz smoothness.

  ## Warning: Computational Cost

  These functions require computing gradients (and potentially gradients of gradients),
  which significantly increases computational cost. Use sparingly:
  - Apply every N training steps instead of every step
  - Use `output_magnitude_penalty/2` as a cheaper proxy

  ## Functions

  - `gradient_penalty/3` - Full gradient norm penalty (expensive)
  - `output_magnitude_penalty/2` - Cheaper proxy penalizing output magnitudes
  """

  import Nx.Defn

  @doc """
  Gradient penalty: penalizes (||∇f|| - target)².

  Computes the gradient of a loss function w.r.t. input tensor,
  then penalizes deviation from target norm.

  ## Options

    * `:target_norm` - Target gradient L2 norm. Default: `1.0`

  ## Example

      # Penalize gradients that deviate from norm 1
      penalty = GradientPenalty.gradient_penalty(
        fn x -> Nx.sum(x) end,
        tensor,
        target_norm: 1.0
      )

  ## Note

  This computes gradients via `Nx.Defn.grad/1`, adding significant overhead.
  Consider using `output_magnitude_penalty/2` for a cheaper alternative.
  """
  @spec gradient_penalty((Nx.Tensor.t() -> Nx.Tensor.t()), Nx.Tensor.t(), keyword()) ::
          Nx.Tensor.t()
  def gradient_penalty(loss_fn, tensor, opts \\ []) do
    target_norm = Keyword.get(opts, :target_norm, 1.0)

    # Compute gradient of loss_fn w.r.t. tensor
    grad_fn = Nx.Defn.grad(loss_fn)
    gradients = grad_fn.(tensor)

    # Compute L2 norm of gradients
    grad_norm =
      gradients
      |> Nx.flatten()
      |> Nx.pow(2)
      |> Nx.sum()
      |> Nx.sqrt()

    # Penalty: (||grad|| - target)²
    target = Nx.tensor(target_norm, type: Nx.type(grad_norm))
    Nx.pow(Nx.subtract(grad_norm, target), 2)
  end

  @doc """
  Output magnitude penalty: cheaper proxy for gradient penalty.

  Penalizes deviation of output magnitude from target, which indirectly
  encourages bounded gradients without computing actual gradients.

  penalty = (||output||₂ - target)²

  ## Options

    * `:target` - Target output L2 norm. Default: `1.0`
    * `:reduction` - `:sum` or `:mean`. Default: `:mean`

  ## Example

      penalty = GradientPenalty.output_magnitude_penalty(model_output, target: 1.0)
  """
  @spec output_magnitude_penalty(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  deftransform output_magnitude_penalty(output, opts \\ []) do
    target = Keyword.get(opts, :target, 1.0)
    reduction = Keyword.get(opts, :reduction, :mean)

    case reduction do
      :mean -> output_magnitude_mean_impl(output, target)
      :sum -> output_magnitude_sum_impl(output, target)
    end
  end

  defnp output_magnitude_mean_impl(output, target) do
    magnitude = Nx.sqrt(Nx.mean(Nx.pow(output, 2)))
    Nx.pow(Nx.subtract(magnitude, target), 2)
  end

  defnp output_magnitude_sum_impl(output, target) do
    magnitude = Nx.sqrt(Nx.sum(Nx.pow(output, 2)))
    Nx.pow(Nx.subtract(magnitude, target), 2)
  end

  @doc """
  WGAN-GP style interpolated gradient penalty.

  Computes gradient penalty at interpolated points between two distributions.

  ## Options

    * `:target_norm` - Target gradient norm. Default: `1.0`

  ## Note

  Requires reference tensor (e.g., from base model or real data).
  """
  @spec interpolated_gradient_penalty(
          (Nx.Tensor.t() -> Nx.Tensor.t()),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          keyword()
        ) :: Nx.Tensor.t()
  def interpolated_gradient_penalty(loss_fn, tensor, reference, opts \\ []) do
    target_norm = Keyword.get(opts, :target_norm, 1.0)

    # Random interpolation coefficient
    key = Nx.Random.key(System.unique_integer([:positive]))
    {epsilon, _} = Nx.Random.uniform(key, shape: {1})

    # Interpolate between tensor and reference
    interpolated =
      Nx.add(
        Nx.multiply(epsilon, tensor),
        Nx.multiply(Nx.subtract(1.0, epsilon), reference)
      )

    # Compute gradient at interpolated point
    gradient_penalty(loss_fn, interpolated, target_norm: target_norm)
  end
end
