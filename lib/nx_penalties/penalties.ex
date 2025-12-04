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
        base_loss = Nx.mean(Nx.pow(Nx.subtract(y_true, y_pred), 2))
        l2_penalty = NxPenalties.Penalties.l2(params, lambda: 0.01)
        Nx.add(base_loss, l2_penalty)
      end
  """

  import Nx.Defn

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
  deftransform l1(tensor, opts \\ []) do
    lambda = Keyword.get(opts, :lambda, 1.0)
    reduction = Keyword.get(opts, :reduction, :sum)

    case reduction do
      :sum -> l1_sum_impl(tensor, lambda)
      :mean -> l1_mean_impl(tensor, lambda)
    end
  end

  defnp l1_sum_impl(tensor, lambda) do
    Nx.multiply(Nx.sum(Nx.abs(tensor)), lambda)
  end

  defnp l1_mean_impl(tensor, lambda) do
    Nx.multiply(Nx.mean(Nx.abs(tensor)), lambda)
  end

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
    * `:center` - Reference value for centering. Default: `nil` (no centering)
      * `nil` - No centering (compute Σx²)
      * `:mean` - Center around tensor mean (compute Σ(x - mean(x))²)
      * `number` - Center around specific value (compute Σ(x - value)²)
      Centering happens before clipping.

  ## Examples

      iex> tensor = Nx.tensor([1.0, 2.0, 3.0])
      iex> NxPenalties.Penalties.l2(tensor, lambda: 0.1)
      #Nx.Tensor<
        f32
        1.4
      >

      iex> tensor = Nx.tensor([1.0, 2.0, 3.0])
      iex> NxPenalties.Penalties.l2(tensor, lambda: 1.0, center: :mean)
      #Nx.Tensor<
        f32
        2.0
      >

  ## Gradient

  The gradient is linear: ∂L2/∂x = 2λx (or 2λ(x - center) when centered)
  """
  @spec l2(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  deftransform l2(tensor, opts \\ []) do
    lambda = Keyword.get(opts, :lambda, 1.0)
    reduction = Keyword.get(opts, :reduction, :sum)
    clip_val = Keyword.get(opts, :clip, nil)
    center = Keyword.get(opts, :center, nil)

    case {reduction, clip_val, center} do
      # No centering, no clip
      {:sum, nil, nil} ->
        l2_sum_impl(tensor, lambda)

      {:mean, nil, nil} ->
        l2_mean_impl(tensor, lambda)

      # No centering, with clip
      {:sum, clip, nil} ->
        l2_sum_clip_impl(tensor, lambda, clip)

      {:mean, clip, nil} ->
        l2_mean_clip_impl(tensor, lambda, clip)

      # Center around mean, no clip
      {:sum, nil, :mean} ->
        l2_sum_center_mean_impl(tensor, lambda)

      {:mean, nil, :mean} ->
        l2_mean_center_mean_impl(tensor, lambda)

      # Center around mean, with clip
      {:sum, clip, :mean} ->
        l2_sum_center_mean_clip_impl(tensor, lambda, clip)

      {:mean, clip, :mean} ->
        l2_mean_center_mean_clip_impl(tensor, lambda, clip)

      # Center around value, no clip
      {:sum, nil, center_val} ->
        l2_sum_center_value_impl(tensor, lambda, center_val)

      {:mean, nil, center_val} ->
        l2_mean_center_value_impl(tensor, lambda, center_val)

      # Center around value, with clip
      {:sum, clip, center_val} ->
        l2_sum_center_value_clip_impl(tensor, lambda, center_val, clip)

      {:mean, clip, center_val} ->
        l2_mean_center_value_clip_impl(tensor, lambda, center_val, clip)
    end
  end

  defnp l2_sum_impl(tensor, lambda) do
    Nx.multiply(Nx.sum(Nx.pow(tensor, 2)), lambda)
  end

  defnp l2_mean_impl(tensor, lambda) do
    Nx.multiply(Nx.mean(Nx.pow(tensor, 2)), lambda)
  end

  defnp l2_sum_clip_impl(tensor, lambda, clip_val) do
    clipped = Nx.clip(tensor, Nx.negate(clip_val), clip_val)
    Nx.multiply(Nx.sum(Nx.pow(clipped, 2)), lambda)
  end

  defnp l2_mean_clip_impl(tensor, lambda, clip_val) do
    clipped = Nx.clip(tensor, Nx.negate(clip_val), clip_val)
    Nx.multiply(Nx.mean(Nx.pow(clipped, 2)), lambda)
  end

  # Center around mean implementations
  defnp l2_sum_center_mean_impl(tensor, lambda) do
    mean = Nx.mean(tensor)
    centered = Nx.subtract(tensor, mean)
    Nx.multiply(Nx.sum(Nx.pow(centered, 2)), lambda)
  end

  defnp l2_mean_center_mean_impl(tensor, lambda) do
    mean = Nx.mean(tensor)
    centered = Nx.subtract(tensor, mean)
    Nx.multiply(Nx.mean(Nx.pow(centered, 2)), lambda)
  end

  defnp l2_sum_center_mean_clip_impl(tensor, lambda, clip_val) do
    mean = Nx.mean(tensor)
    centered = Nx.subtract(tensor, mean)
    clipped = Nx.clip(centered, Nx.negate(clip_val), clip_val)
    Nx.multiply(Nx.sum(Nx.pow(clipped, 2)), lambda)
  end

  defnp l2_mean_center_mean_clip_impl(tensor, lambda, clip_val) do
    mean = Nx.mean(tensor)
    centered = Nx.subtract(tensor, mean)
    clipped = Nx.clip(centered, Nx.negate(clip_val), clip_val)
    Nx.multiply(Nx.mean(Nx.pow(clipped, 2)), lambda)
  end

  # Center around value implementations
  defnp l2_sum_center_value_impl(tensor, lambda, center_value) do
    centered = Nx.subtract(tensor, center_value)
    Nx.multiply(Nx.sum(Nx.pow(centered, 2)), lambda)
  end

  defnp l2_mean_center_value_impl(tensor, lambda, center_value) do
    centered = Nx.subtract(tensor, center_value)
    Nx.multiply(Nx.mean(Nx.pow(centered, 2)), lambda)
  end

  defnp l2_sum_center_value_clip_impl(tensor, lambda, center_value, clip_val) do
    centered = Nx.subtract(tensor, center_value)
    clipped = Nx.clip(centered, Nx.negate(clip_val), clip_val)
    Nx.multiply(Nx.sum(Nx.pow(clipped, 2)), lambda)
  end

  defnp l2_mean_center_value_clip_impl(tensor, lambda, center_value, clip_val) do
    centered = Nx.subtract(tensor, center_value)
    clipped = Nx.clip(centered, Nx.negate(clip_val), clip_val)
    Nx.multiply(Nx.mean(Nx.pow(clipped, 2)), lambda)
  end

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
  deftransform elastic_net(tensor, opts \\ []) do
    lambda = Keyword.get(opts, :lambda, 1.0)
    l1_ratio = Keyword.get(opts, :l1_ratio, 0.5)
    reduction = Keyword.get(opts, :reduction, :sum)

    case reduction do
      :sum -> elastic_net_sum_impl(tensor, lambda, l1_ratio)
      :mean -> elastic_net_mean_impl(tensor, lambda, l1_ratio)
    end
  end

  defnp elastic_net_sum_impl(tensor, lambda, l1_ratio) do
    l1_raw = Nx.sum(Nx.abs(tensor))
    l2_raw = Nx.sum(Nx.pow(tensor, 2))

    l1_weighted = Nx.multiply(l1_raw, l1_ratio)
    l2_weighted = Nx.multiply(l2_raw, Nx.subtract(1.0, l1_ratio))
    combined = Nx.add(l1_weighted, l2_weighted)

    Nx.multiply(combined, lambda)
  end

  defnp elastic_net_mean_impl(tensor, lambda, l1_ratio) do
    l1_raw = Nx.mean(Nx.abs(tensor))
    l2_raw = Nx.mean(Nx.pow(tensor, 2))

    l1_weighted = Nx.multiply(l1_raw, l1_ratio)
    l2_weighted = Nx.multiply(l2_raw, Nx.subtract(1.0, l1_ratio))
    combined = Nx.add(l1_weighted, l2_weighted)

    Nx.multiply(combined, lambda)
  end
end
