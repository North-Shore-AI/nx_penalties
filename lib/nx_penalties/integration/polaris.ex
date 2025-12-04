defmodule NxPenalties.Integration.Polaris do
  @moduledoc """
  Gradient transformations for use with Polaris optimizers.

  These transforms operate on gradients and parameters, not on
  the loss function. They follow Polaris's composable pattern.

  ## Gradient-Level vs Loss-Level Regularization

  There are two ways to apply weight decay:

  1. **Loss-based** (NxPenalties default): Add L2 penalty to loss
     - `loss_total = loss + λ * ||w||²`
     - Gradient: `∂loss/∂w + 2λw`

  2. **Gradient transform** (Polaris style): Modify gradients directly
     - `grad_new = grad + λw`
     - Equivalent to AdamW-style decoupled weight decay

  This module provides gradient transforms for Polaris-style regularization.

  ## Composition

  Polaris transforms compose via piping:

      optimizer =
        Polaris.Optimizers.adam(learning_rate: 0.001)
        |> NxPenalties.Integration.Polaris.add_l2_decay(0.01)
        |> NxPenalties.Integration.Polaris.add_l1_decay(0.001)

  ## Weight Decay vs L2 Regularization

  These are mathematically equivalent for SGD but differ for
  adaptive optimizers like Adam. Weight decay (implemented here)
  is generally preferred for modern training.

  Reference: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017)
  """

  @doc """
  Add L2 weight decay to gradients.

  This is equivalent to Polaris's built-in weight decay in AdamW,
  provided for completeness and explicit composition.

  Weight decay modifies the gradient: g' = g + λw
  Where λ is the decay rate and w is the weight.

  ## Parameters

    * `optimizer` - Polaris optimizer tuple `{init_fn, update_fn}`
    * `decay` - Decay rate. Default: `0.01`

  ## Note

  For AdamW, prefer using the built-in `:decay` option:

      Polaris.Optimizers.adamw(learning_rate: 0.001, decay: 0.01)

  This transform is useful when you want to add decay to an optimizer
  that doesn't have it built-in, or for explicit composition.

  ## Example

      optimizer =
        Polaris.Optimizers.adam(learning_rate: 0.001)
        |> NxPenalties.Integration.Polaris.add_l2_decay(0.01)
  """
  @spec add_l2_decay(term(), float()) :: term()
  def add_l2_decay(optimizer, decay \\ 0.01) do
    {base_init, base_update} = optimizer

    init_fn = fn params ->
      base_state = base_init.(params)
      %{base: base_state, l2_decay: decay}
    end

    update_fn = fn gradients, state, params ->
      # Add decay to gradients: g' = g + λw
      decayed_grads =
        deep_map(gradients, params, fn g, w ->
          Nx.add(g, Nx.multiply(w, state.l2_decay))
        end)

      # Apply base optimizer
      {updates, new_base_state} = base_update.(decayed_grads, state.base, params)

      new_state = %{state | base: new_base_state}
      {updates, new_state}
    end

    {init_fn, update_fn}
  end

  @doc """
  Add L1 weight decay (sign decay) to gradients.

  Modifies the gradient: g' = g + λ * sign(w)
  This encourages sparsity in the weights.

  ## Parameters

    * `optimizer` - Polaris optimizer tuple
    * `decay` - Decay rate. Default: `0.001`

  ## Note

  L1 decay can cause weights to oscillate around zero. Consider
  using a small threshold to zero out very small weights.

  ## Example

      optimizer =
        Polaris.Optimizers.sgd(learning_rate: 0.01)
        |> NxPenalties.Integration.Polaris.add_l1_decay(0.001)
  """
  @spec add_l1_decay(term(), float()) :: term()
  def add_l1_decay(optimizer, decay \\ 0.001) do
    {base_init, base_update} = optimizer

    init_fn = fn params ->
      base_state = base_init.(params)
      %{base: base_state, l1_decay: decay}
    end

    update_fn = fn gradients, state, params ->
      # Add L1 decay: g' = g + λ * sign(w)
      decayed_grads =
        deep_map(gradients, params, fn g, w ->
          Nx.add(g, Nx.multiply(Nx.sign(w), state.l1_decay))
        end)

      {updates, new_base_state} = base_update.(decayed_grads, state.base, params)

      new_state = %{state | base: new_base_state}
      {updates, new_state}
    end

    {init_fn, update_fn}
  end

  @doc """
  Add elastic net (L1 + L2) weight decay to gradients.

  Combines L1 and L2 decay:
  g' = g + λ * (α * sign(w) + (1-α) * w)

  ## Parameters

    * `optimizer` - Polaris optimizer tuple
    * `decay` - Overall decay rate. Default: `0.01`
    * `l1_ratio` - Ratio of L1 to L2 (α). Default: `0.5`

  ## Example

      optimizer =
        Polaris.Optimizers.adam(learning_rate: 0.001)
        |> NxPenalties.Integration.Polaris.add_elastic_net_decay(0.01, 0.3)
  """
  @spec add_elastic_net_decay(term(), float(), float()) :: term()
  def add_elastic_net_decay(optimizer, decay \\ 0.01, l1_ratio \\ 0.5) do
    {base_init, base_update} = optimizer

    init_fn = fn params ->
      base_state = base_init.(params)
      %{base: base_state, decay: decay, l1_ratio: l1_ratio}
    end

    update_fn = fn gradients, state, params ->
      alpha = state.l1_ratio
      lambda = state.decay

      decayed_grads =
        deep_map(gradients, params, fn g, w ->
          l1_term = Nx.multiply(Nx.sign(w), alpha)
          l2_term = Nx.multiply(w, 1.0 - alpha)
          decay_term = Nx.multiply(Nx.add(l1_term, l2_term), lambda)
          Nx.add(g, decay_term)
        end)

      {updates, new_base_state} = base_update.(decayed_grads, state.base, params)

      new_state = %{state | base: new_base_state}
      {updates, new_state}
    end

    {init_fn, update_fn}
  end

  # Deep map over two nested structures (gradients and params)
  # Check for tensors first since Nx.Tensor is a struct (which is also a map)
  defp deep_map(%Nx.Tensor{} = gradient, %Nx.Tensor{} = param, fun) do
    fun.(gradient, param)
  end

  defp deep_map(gradients, params, fun) when is_map(gradients) and is_map(params) do
    Map.new(gradients, fn {key, g} ->
      p = Map.fetch!(params, key)
      {key, deep_map(g, p, fun)}
    end)
  end

  defp deep_map(gradient, param, fun) do
    fun.(gradient, param)
  end
end
