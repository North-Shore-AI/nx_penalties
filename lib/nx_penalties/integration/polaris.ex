defmodule NxPenalties.Integration.Polaris do
  @moduledoc """
  Integration with Polaris gradient transforms.

  > **v0.2 Preview**: Stub implementation. Full gradient-level
  > regularization (weight decay transforms) coming in v0.2.

  ## Gradient-Level vs Loss-Level Regularization

  There are two ways to apply weight decay:

  1. **Loss-based** (NxPenalties default): Add L2 penalty to loss
     - `loss_total = loss + λ * ||w||²`
     - Gradient: `∂loss/∂w + 2λw`

  2. **Gradient transform** (Polaris style): Modify gradients directly
     - `grad_new = grad + λw`
     - Equivalent to AdamW-style decoupled weight decay

  This module provides gradient transforms for Polaris-style regularization.

  ## Example (v0.2)

      optimizer =
        Polaris.Optimizers.adam(learning_rate: 0.001)
        |> NxPenalties.Integration.Polaris.add_l2_decay(0.01)
  """

  @doc """
  Add L2 weight decay as a gradient transform.

  > **v0.2**: Stub implementation - returns optimizer unchanged.

  ## Parameters

    * `optimizer` - Polaris optimizer tuple `{init_fn, update_fn}`
    * `decay` - Weight decay coefficient

  ## Returns

  Modified optimizer tuple.
  """
  @spec add_l2_decay(term(), float()) :: term()
  def add_l2_decay(optimizer, _decay) do
    # v0.2: Full implementation
    # For now, return unchanged
    optimizer
  end

  @doc """
  Add L1 weight decay as a gradient transform.

  > **v0.2**: Stub implementation - returns optimizer unchanged.

  ## Parameters

    * `optimizer` - Polaris optimizer tuple
    * `decay` - L1 decay coefficient
  """
  @spec add_l1_decay(term(), float()) :: term()
  def add_l1_decay(optimizer, _decay) do
    # v0.2: Full implementation
    optimizer
  end

  @doc """
  Add elastic net decay as a gradient transform.

  > **v0.2**: Stub implementation - returns optimizer unchanged.

  ## Parameters

    * `optimizer` - Polaris optimizer tuple
    * `decay` - Total decay coefficient
    * `l1_ratio` - Balance between L1 and L2 (0.0 to 1.0)
  """
  @spec add_elastic_net_decay(term(), float(), float()) :: term()
  def add_elastic_net_decay(optimizer, _decay, _l1_ratio \\ 0.5) do
    # v0.2: Full implementation
    optimizer
  end
end
