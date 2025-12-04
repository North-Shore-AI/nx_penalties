defmodule NxPenalties.Integration.Axon do
  @moduledoc """
  Helpers for integrating NxPenalties with Axon training loops.

  Since Axon explicitly rejects model-level regularization ("regularization
  is a concern of training/optimization and not the model"), we provide
  training loop helpers rather than layer modifications.

  ## Integration Patterns

  ### Pattern 1: Wrap Loss Function

  The simplest approach - wrap your loss function with penalties:

      loss_fn = NxPenalties.Integration.Axon.wrap_loss(
        &Axon.Losses.mean_squared_error/2,
        &NxPenalties.Penalties.l2/2,
        lambda: 0.01
      )

      Axon.Loop.trainer(model, loss_fn, optimizer)

  ### Pattern 2: Pipeline-Based Loss

  For multiple penalties with individual weights:

      pipeline = NxPenalties.pipeline([
        {:l2, weight: 0.01},
        {:entropy, weight: 0.001, opts: [mode: :penalty]}
      ])

      loss_fn = NxPenalties.Integration.Axon.wrap_loss_with_pipeline(
        &Axon.Losses.categorical_cross_entropy/2,
        pipeline
      )

  ## Note on Axon Availability

  This module requires Axon to be available. If Axon is not installed,
  the functions will raise at runtime.
  """

  @doc """
  Wrap a loss function to include a single penalty term.

  ## Parameters

    * `base_loss_fn` - Original loss function `(y_true, y_pred) -> scalar`
    * `penalty_fn` - Penalty function `(tensor, opts) -> scalar`
    * `opts` - Options:
      * `:lambda` - Weight for penalty term. Default: `0.01`
      * `:penalty_opts` - Options passed to penalty function

  ## Returns

  A new loss function with signature `(y_true, y_pred) -> scalar`

  ## Example

      wrapped_loss = NxPenalties.Integration.Axon.wrap_loss(
        &Axon.Losses.mean_squared_error/2,
        &NxPenalties.Penalties.l2/2,
        lambda: 0.01
      )

      model
      |> Axon.Loop.trainer(wrapped_loss, optimizer)
      |> Axon.Loop.run(data, %{}, epochs: 10)
  """
  @spec wrap_loss(function(), function(), keyword()) :: function()
  def wrap_loss(base_loss_fn, penalty_fn, opts \\ []) do
    lambda = Keyword.get(opts, :lambda, 0.01)
    penalty_opts = Keyword.get(opts, :penalty_opts, [])

    fn y_true, y_pred ->
      base_loss = base_loss_fn.(y_true, y_pred)
      penalty = penalty_fn.(y_pred, penalty_opts)
      Nx.add(base_loss, Nx.multiply(penalty, lambda))
    end
  end

  @doc """
  Wrap a loss function with a full penalty pipeline.

  More flexible than `wrap_loss/3` - supports multiple penalties
  with individual weights.

  ## Parameters

    * `base_loss_fn` - Original loss function
    * `pipeline` - `NxPenalties.Pipeline` struct
    * `opts` - Additional options

  ## Returns

  A new loss function. Note: metrics from pipeline are not accessible
  with this pattern. Use `build_train_step/4` for metrics.

  ## Example

      pipeline =
        NxPenalties.Pipeline.new()
        |> NxPenalties.Pipeline.add(:l1, &NxPenalties.Penalties.l1/2, weight: 0.001)
        |> NxPenalties.Pipeline.add(:entropy, &NxPenalties.Divergences.entropy/2,
             weight: 0.01, opts: [mode: :penalty])

      wrapped_loss = NxPenalties.Integration.Axon.wrap_loss_with_pipeline(
        &Axon.Losses.categorical_cross_entropy/2,
        pipeline
      )
  """
  @spec wrap_loss_with_pipeline(function(), NxPenalties.Pipeline.t(), keyword()) :: function()
  def wrap_loss_with_pipeline(base_loss_fn, pipeline, opts \\ []) do
    fn y_true, y_pred ->
      base_loss = base_loss_fn.(y_true, y_pred)
      penalty_total = NxPenalties.Pipeline.compute_total(pipeline, y_pred, opts)
      Nx.add(base_loss, penalty_total)
    end
  end

  @doc """
  Create a loss wrapper that applies penalties to model parameters.

  This is useful for weight decay on the model parameters themselves,
  rather than on the predictions.

  ## Parameters

    * `base_loss_fn` - Original loss function `(y_true, y_pred) -> scalar`
    * `param_penalty_fn` - Function `(params) -> scalar` that computes penalty on params
    * `opts` - Options:
      * `:lambda` - Weight for penalty. Default: `0.01`

  ## Example

      # L2 weight decay on all parameters
      param_penalty = fn params ->
        params
        |> Nx.Defn.Tree.flatten()
        |> Enum.map(&NxPenalties.Penalties.l2(&1, lambda: 1.0))
        |> Enum.reduce(Nx.tensor(0.0), &Nx.add/2)
      end

      loss_with_decay = NxPenalties.Integration.Axon.wrap_loss_with_params(
        &Axon.Losses.mean_squared_error/2,
        param_penalty,
        lambda: 0.001
      )
  """
  @spec wrap_loss_with_params(function(), function(), keyword()) :: function()
  def wrap_loss_with_params(base_loss_fn, param_penalty_fn, opts \\ []) do
    lambda = Keyword.get(opts, :lambda, 0.01)

    fn y_true, y_pred, params ->
      base_loss = base_loss_fn.(y_true, y_pred)
      param_penalty = param_penalty_fn.(params)
      Nx.add(base_loss, Nx.multiply(param_penalty, lambda))
    end
  end
end
