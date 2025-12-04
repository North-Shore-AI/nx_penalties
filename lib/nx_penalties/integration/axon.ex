if Code.ensure_loaded?(Axon) do
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

    ### Pattern 3: Custom Training Step

    For full control with metrics access:

        {init_fn, step_fn} = NxPenalties.Integration.Axon.build_train_step(
          model,
          &Axon.Losses.mean_squared_error/2,
          pipeline,
          Polaris.Optimizers.adam(learning_rate: 0.001)
        )

    ### Pattern 4: Activity Regularization (Advanced)

    Capture intermediate activations for penalties:

        model =
          Axon.input("input")
          |> Axon.dense(128)
          |> NxPenalties.Integration.Axon.capture_activation(:hidden1)
          |> Axon.dense(10)

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

    @doc """
    Build a complete training step function with penalty support.

    This pattern provides full control and access to metrics from
    each penalty computation.

    ## Parameters

      * `model` - Axon model
      * `base_loss_fn` - Loss function `(y_true, y_pred) -> scalar`
      * `pipeline` - `NxPenalties.Pipeline` struct
      * `optimizer` - Polaris optimizer tuple `{init_fn, update_fn}`

    ## Returns

    A tuple `{init_fn, step_fn}` compatible with custom training loops.

    ## Metrics

    The step function returns these metrics:
    - `"base_loss"` - Task loss before penalties
    - `"penalty_total"` - Sum of all penalties
    - `"loss"` - Total loss (base + penalties)
    - Plus individual penalty metrics from pipeline

    ## Example

        pipeline = NxPenalties.pipeline([
          {:l2, weight: 0.01},
          {:entropy, weight: 0.001, opts: [mode: :penalty]}
        ])

        {init_fn, step_fn} = NxPenalties.Integration.Axon.build_train_step(
          model,
          &Axon.Losses.mean_squared_error/2,
          pipeline,
          Polaris.Optimizers.adam(learning_rate: 0.001)
        )

        # Initialize state
        state = init_fn.(Nx.template({1, 10}, :f32), %{})

        # Run training step
        {new_state, metrics} = step_fn.(state, {x_batch, y_batch})
    """
    @spec build_train_step(term(), function(), NxPenalties.Pipeline.t(), term()) ::
            {function(), function()}
    def build_train_step(model, base_loss_fn, pipeline, optimizer) do
      # Build in inference mode to get prediction-only output
      {model_init_fn, predict_fn} = Axon.build(model)
      {opt_init_fn, opt_update_fn} = optimizer

      # Initialize function
      init_fn = fn template, init_state ->
        model_state = model_init_fn.(template, init_state)
        params = Axon.ModelState.trainable_parameters(model_state)
        opt_state = opt_init_fn.(params)

        %{
          model_state: model_state,
          optimizer_state: opt_state,
          step: 0
        }
      end

      # Step function
      step_fn = fn state, batch ->
        %{model_state: model_state, optimizer_state: opt_state} = state
        {x, y_true} = batch

        params = Axon.ModelState.trainable_parameters(model_state)

        # Compute gradients
        {loss, grads} =
          Nx.Defn.value_and_grad(params, fn p ->
            temp_state = Axon.ModelState.update(model_state, p)
            y_pred = predict_fn.(temp_state, x)

            # Base loss
            base_loss = base_loss_fn.(y_true, y_pred)

            # Penalties on predictions
            penalty_total = NxPenalties.Pipeline.compute_total(pipeline, y_pred)

            # Total loss for gradient
            Nx.add(base_loss, penalty_total)
          end)

        # Compute metrics (without grad tracking)
        temp_state = Axon.ModelState.update(model_state, params)
        y_pred = predict_fn.(temp_state, x) |> Nx.backend_transfer()
        base_loss = base_loss_fn.(y_true, y_pred)
        {penalty_total, penalty_metrics} = NxPenalties.Pipeline.compute(pipeline, y_pred)

        # Update parameters
        {updates, new_opt_state} = opt_update_fn.(grads, opt_state, params)
        new_params = apply_updates(params, updates)
        new_model_state = Axon.ModelState.update(model_state, new_params)

        # Build metrics
        metrics =
          Map.merge(penalty_metrics, %{
            "base_loss" => Nx.to_number(base_loss),
            "penalty_total" => Nx.to_number(penalty_total),
            "loss" => Nx.to_number(loss)
          })

        {%{model_state: new_model_state, optimizer_state: new_opt_state}, metrics}
      end

      {init_fn, step_fn}
    end

    @doc """
    Build training step with weight regularization on model parameters.

    Unlike `build_train_step/4` which applies penalties to predictions,
    this applies penalties directly to the model weights (true weight decay).

    ## Parameters

      * `model` - Axon model
      * `base_loss_fn` - Loss function `(y_true, y_pred) -> scalar`
      * `param_penalty_fn` - Function `(params) -> scalar` for weight penalty
      * `optimizer` - Polaris optimizer tuple
      * `opts` - Options:
        * `:lambda` - Weight for parameter penalty. Default: `0.01`

    ## Example

        # L2 weight decay on all parameters
        param_penalty = fn params ->
          params
          |> flatten_params()
          |> Enum.map(&NxPenalties.l2(&1, lambda: 1.0))
          |> Enum.reduce(Nx.tensor(0.0), &Nx.add/2)
        end

        {init_fn, step_fn} = build_train_step_with_weight_decay(
          model,
          &Axon.Losses.mean_squared_error/2,
          param_penalty,
          optimizer,
          lambda: 0.001
        )
    """
    @spec build_train_step_with_weight_decay(term(), function(), function(), term(), keyword()) ::
            {function(), function()}
    def build_train_step_with_weight_decay(
          model,
          base_loss_fn,
          param_penalty_fn,
          optimizer,
          opts \\ []
        ) do
      lambda = Keyword.get(opts, :lambda, 0.01)
      {model_init_fn, predict_fn} = Axon.build(model)
      {opt_init_fn, opt_update_fn} = optimizer

      init_fn = fn template, init_state ->
        model_state = model_init_fn.(template, init_state)
        params = Axon.ModelState.trainable_parameters(model_state)
        opt_state = opt_init_fn.(params)

        %{
          model_state: model_state,
          optimizer_state: opt_state,
          step: 0
        }
      end

      step_fn = fn state, batch ->
        %{model_state: model_state, optimizer_state: opt_state} = state
        {x, y_true} = batch

        params = Axon.ModelState.trainable_parameters(model_state)

        {loss, grads} =
          Nx.Defn.value_and_grad(params, fn p ->
            temp_state = Axon.ModelState.update(model_state, p)
            y_pred = predict_fn.(temp_state, x)

            base_loss = base_loss_fn.(y_true, y_pred)
            weight_penalty = Nx.multiply(param_penalty_fn.(p), lambda)

            Nx.add(base_loss, weight_penalty)
          end)

        # Compute metrics
        temp_state = Axon.ModelState.update(model_state, params)
        y_pred = predict_fn.(temp_state, x) |> Nx.backend_transfer()
        base_loss = base_loss_fn.(y_true, y_pred)
        weight_penalty = Nx.multiply(param_penalty_fn.(params), lambda)

        {updates, new_opt_state} = opt_update_fn.(grads, opt_state, params)
        new_params = apply_updates(params, updates)
        new_model_state = Axon.ModelState.update(model_state, new_params)

        metrics = %{
          "base_loss" => Nx.to_number(base_loss),
          "weight_penalty" => Nx.to_number(weight_penalty),
          "loss" => Nx.to_number(loss)
        }

        {%{model_state: new_model_state, optimizer_state: new_opt_state}, metrics}
      end

      {init_fn, step_fn}
    end

    @doc """
    Insert a capture layer that stores activations in model state.

    The captured activations can be extracted after the forward pass
    and used for activity regularization.

    ## Parameters

      * `model` - Axon model (at desired capture point)
      * `name` - Atom name for this capture point

    ## Example

        model =
          Axon.input("input", shape: {nil, 784})
          |> Axon.dense(256)
          |> NxPenalties.Integration.Axon.capture_activation(:hidden1)
          |> Axon.dense(128)
          |> NxPenalties.Integration.Axon.capture_activation(:hidden2)
          |> Axon.dense(10)
    """
    @spec capture_activation(Axon.t(), atom()) :: Axon.t()
    def capture_activation(model, name) do
      Axon.layer(
        fn input, _state ->
          %Axon.StatefulOutput{
            output: input,
            state: %{Atom.to_string(name) => input}
          }
        end,
        [model],
        name: "capture_#{name}"
      )
    end

    @doc """
    Extract captured activations from model state after forward pass.
    """
    @spec extract_captures(map()) :: map()
    def extract_captures(model_state) do
      model_state
      |> Enum.flat_map(fn
        {_layer_name, captures} when is_map(captures) ->
          Enum.map(captures, fn {key, value} ->
            atom_key = if is_binary(key), do: String.to_atom(key), else: key
            {atom_key, value}
          end)

        _ ->
          []
      end)
      |> Map.new()
    end

    @doc """
    Build training step with activity regularization on captured layers.

    ## Parameters

      * `model` - Axon model with capture layers
      * `base_loss_fn` - Task loss function
      * `activity_penalties` - Map of capture name to `{penalty_fn, opts}` tuples
      * `optimizer` - Polaris optimizer
    """
    @spec build_activity_train_step(term(), function(), map(), term()) :: {function(), function()}
    def build_activity_train_step(model, base_loss_fn, activity_penalties, optimizer) do
      {model_init_fn, predict_fn} = Axon.build(model, mode: :train)
      {opt_init_fn, opt_update_fn} = optimizer

      init = fn template, init_state ->
        model_state = model_init_fn.(template, init_state)
        params = Axon.ModelState.trainable_parameters(model_state)
        opt_state = opt_init_fn.(params)
        %{model_state: model_state, optimizer_state: opt_state}
      end

      activity_loss_fn = fn captures ->
        activity_penalties
        |> Enum.map(fn {name, {penalty_fn, opts}} ->
          case Map.get(captures, name) do
            nil ->
              Nx.tensor(0.0)

            activation ->
              weight = Keyword.get(opts, :weight, 1.0)
              penalty_opts = Keyword.delete(opts, :weight)
              Nx.multiply(penalty_fn.(activation, penalty_opts), weight)
          end
        end)
        |> Enum.reduce(Nx.tensor(0.0), &Nx.add/2)
      end

      step = fn state, batch ->
        %{model_state: model_state, optimizer_state: opt_state} = state
        {x, y_true} = batch
        params = Axon.ModelState.trainable_parameters(model_state)

        {loss, grads} =
          Nx.Defn.value_and_grad(params, fn p ->
            temp_state = Axon.ModelState.update(model_state, p)
            %{prediction: y_pred, state: forward_state} = predict_fn.(temp_state, x)

            base_loss = base_loss_fn.(y_true, y_pred)
            activity_loss = activity_loss_fn.(extract_captures(forward_state))

            Nx.add(base_loss, activity_loss)
          end)

        # Metrics (non-grad)
        temp_state = Axon.ModelState.update(model_state, params)
        %{prediction: y_pred, state: forward_state} = predict_fn.(temp_state, x)
        y_pred = Nx.backend_transfer(y_pred)
        base_loss = base_loss_fn.(y_true, y_pred)
        activity_loss = activity_loss_fn.(extract_captures(forward_state))

        {updates, new_opt_state} = opt_update_fn.(grads, opt_state, params)
        new_params = apply_updates(params, updates)
        new_model_state = Axon.ModelState.update(model_state, new_params)

        metrics = %{
          "loss" => Nx.to_number(loss),
          "base_loss" => Nx.to_number(base_loss),
          "activity_loss" => Nx.to_number(activity_loss)
        }

        {%{model_state: new_model_state, optimizer_state: new_opt_state}, metrics}
      end

      {init, step}
    end

    @doc """
    Create a weight schedule function for curriculum learning.

    Returns a function that computes penalty weights based on the current epoch.

    ## Parameters

      * `schedules` - Map of penalty name to schedule configuration

    ## Schedule Types

      * `{:linear, start, stop}` - Linear interpolation from start to stop
      * `{:warmup, final_value, warmup_epochs}` - Ramp up to final value
      * `{:decay, initial, decay_rate}` - Exponential decay
      * `{:constant, value}` - Fixed value

    ## Example

        schedule_fn = NxPenalties.Integration.Axon.weight_schedule(%{
          l2: {:linear, 0.0, 0.01},        # Ramp L2 from 0 to 0.01
          kl: {:warmup, 0.1, 10},          # Warm up KL over 10 epochs
          entropy: {:constant, 0.001}      # Keep entropy constant
        })

        # Use in training loop
        epoch = 5
        total_epochs = 100
        weights = schedule_fn.(epoch, total_epochs)
        # => %{l2: 0.0005, kl: 0.05, entropy: 0.001}
    """
    @spec weight_schedule(map()) :: function()
    def weight_schedule(schedules) do
      fn epoch, total_epochs ->
        Map.new(schedules, fn {name, schedule} ->
          weight = compute_scheduled_weight(schedule, epoch, total_epochs)
          {name, weight}
        end)
      end
    end

    defp compute_scheduled_weight({:linear, start_val, end_val}, epoch, total_epochs) do
      progress = min(epoch / max(total_epochs - 1, 1), 1.0)
      start_val + (end_val - start_val) * progress
    end

    defp compute_scheduled_weight({:warmup, final_val, warmup_epochs}, epoch, _total_epochs) do
      if epoch >= warmup_epochs do
        final_val
      else
        final_val * (epoch / warmup_epochs)
      end
    end

    defp compute_scheduled_weight({:decay, initial, decay_rate}, epoch, _total_epochs) do
      initial * :math.pow(decay_rate, epoch)
    end

    defp compute_scheduled_weight({:constant, value}, _epoch, _total_epochs) do
      value
    end

    @doc """
    Apply scheduled weights to a pipeline.

    ## Example

        pipeline = NxPenalties.pipeline([
          {:l2, weight: 0.0},
          {:kl, weight: 0.0}
        ])

        schedule_fn = weight_schedule(%{
          l2: {:linear, 0.0, 0.01},
          kl: {:warmup, 0.1, 10}
        })

        # In training loop
        weights = schedule_fn.(current_epoch, total_epochs)
        pipeline = apply_scheduled_weights(pipeline, weights)
    """
    @spec apply_scheduled_weights(NxPenalties.Pipeline.t(), map()) :: NxPenalties.Pipeline.t()
    def apply_scheduled_weights(pipeline, weights) do
      Enum.reduce(weights, pipeline, fn {name, weight}, acc ->
        NxPenalties.Pipeline.update_weight(acc, name, weight)
      end)
    end

    # Helper to apply updates to nested params
    # Tensors are matched first since they are also structs (maps)
    defp apply_updates(%Nx.Tensor{} = param, %Nx.Tensor{} = update) do
      Nx.add(param, update)
    end

    defp apply_updates(%Nx.Tensor{} = param, update) do
      Nx.add(param, update)
    end

    # Handle nested maps (but not tensors which are structs)
    defp apply_updates(%{__struct__: _} = param, _update) do
      # Skip other structs
      param
    end

    defp apply_updates(params, updates) when is_map(params) and is_map(updates) do
      Map.new(params, fn {key, param_val} ->
        case Map.get(updates, key) do
          nil -> {key, param_val}
          update_val -> {key, apply_updates(param_val, update_val)}
        end
      end)
    end

    defp apply_updates(param, _update) when is_binary(param) do
      # Skip binary data (e.g., from model state metadata)
      param
    end

    defp apply_updates(param, update) when is_number(param) and is_number(update) do
      param + update
    end

    defp apply_updates(param, update) do
      # For tensors that aren't explicitly matched
      Nx.add(param, update)
    end

    # ============================================================================
    # Axon.Loop Integration
    # ============================================================================

    @doc """
    Add penalty metrics logging to an Axon training loop.

    This adds metrics for each penalty in the pipeline to be tracked
    during training. The metrics are computed on the model predictions
    at the end of each step.

    ## Parameters

      * `loop` - An `Axon.Loop` struct
      * `pipeline` - `NxPenalties.Pipeline` struct
      * `opts` - Options:
        * `:on` - Event to compute metrics on. Default: `:iteration_completed`

    ## Example

        pipeline = NxPenalties.pipeline([
          {:l2, weight: 0.01},
          {:entropy, weight: 0.001, opts: [mode: :penalty]}
        ])

        model
        |> Axon.Loop.trainer(loss, optimizer)
        |> NxPenalties.Integration.Axon.log_penalties(pipeline)
        |> Axon.Loop.run(data, %{}, epochs: 10)

    ## Metrics Added

    For each penalty in the pipeline:
    - `"{name}"` - Raw penalty value
    - `"{name}_weighted"` - Penalty value after weight applied
    """
    @spec log_penalties(struct(), NxPenalties.Pipeline.t(), keyword()) :: struct()
    def log_penalties(loop, pipeline, opts \\ []) do
      on_event = Keyword.get(opts, :on, :iteration_completed)

      # Extract penalty names from pipeline
      penalty_names =
        pipeline.entries
        |> Enum.filter(fn {_, _, _, _, enabled} -> enabled end)
        |> Enum.map(fn {name, _, _, _, _} -> name end)

      # Add a handler to compute and log penalty metrics
      Axon.Loop.handle_event(loop, on_event, fn state ->
        # Get the last predictions from state if available
        case Map.get(state, :y_pred) do
          nil ->
            {:continue, state}

          y_pred ->
            {_total, metrics} = NxPenalties.Pipeline.compute(pipeline, y_pred)

            # Merge penalty metrics into loop metrics
            updated_metrics =
              Enum.reduce(penalty_names, state.metrics, fn name, acc ->
                name_str = Atom.to_string(name)
                raw_key = name_str
                weighted_key = "#{name_str}_weighted"

                acc
                |> Map.put(raw_key, Map.get(metrics, raw_key, 0.0))
                |> Map.put(weighted_key, Map.get(metrics, weighted_key, 0.0))
              end)

            {:continue, %{state | metrics: updated_metrics}}
        end
      end)
    end

    @doc """
    Add a callback to update pipeline weights during training.

    Useful for curriculum learning or scheduled regularization where
    penalty weights change over the course of training.

    ## Parameters

      * `loop` - An `Axon.Loop` struct
      * `pipeline_key` - Key in loop state where pipeline is stored
      * `schedule_fn` - Function `(epoch, total_epochs) -> weight_map`
      * `opts` - Options:
        * `:total_epochs` - Total number of epochs for schedule. Default: `100`

    ## Example

        schedule_fn = NxPenalties.Integration.Axon.weight_schedule(%{
          l2: {:linear, 0.0, 0.01},
          kl: {:warmup, 0.1, 10}
        })

        model
        |> Axon.Loop.trainer(loss, optimizer)
        |> Axon.Loop.handle_event(:epoch_started, fn state ->
          # Store pipeline in state on first epoch
          pipeline = state[:handler_metadata][:pipeline] || initial_pipeline
          {:continue, put_in(state[:handler_metadata][:pipeline], pipeline)}
        end)
        |> NxPenalties.Integration.Axon.schedule_weights(:pipeline, schedule_fn, total_epochs: 50)
        |> Axon.Loop.run(data, %{}, epochs: 50)

    ## Alternative: Direct Usage

    For simpler cases, you can update weights directly in a handler:

        Axon.Loop.handle_event(loop, :epoch_started, fn state ->
          epoch = state.epoch
          weights = schedule_fn.(epoch, total_epochs)
          updated_pipeline = apply_scheduled_weights(pipeline, weights)
          # Use updated_pipeline in your loss computation
          {:continue, state}
        end)
    """
    @spec schedule_weights(struct(), atom(), function(), keyword()) :: struct()
    def schedule_weights(loop, pipeline_key, schedule_fn, opts \\ []) do
      total_epochs = Keyword.get(opts, :total_epochs, 100)

      Axon.Loop.handle_event(loop, :epoch_started, fn state ->
        epoch = state.epoch

        # Get the current pipeline from handler metadata
        handler_metadata = Map.get(state, :handler_metadata, %{})
        pipeline = Map.get(handler_metadata, pipeline_key)

        if pipeline do
          # Compute scheduled weights for this epoch
          weights = schedule_fn.(epoch, total_epochs)

          # Apply weights to pipeline
          updated_pipeline = apply_scheduled_weights(pipeline, weights)

          # Store updated pipeline back in state
          updated_metadata = Map.put(handler_metadata, pipeline_key, updated_pipeline)
          {:continue, %{state | handler_metadata: updated_metadata}}
        else
          {:continue, state}
        end
      end)
    end

    @doc """
    Create a training loop with integrated penalty support.

    This is a convenience function that combines model building,
    loss computation with penalties, and optimizer setup.

    ## Parameters

      * `model` - Axon model
      * `base_loss_fn` - Base loss function
      * `pipeline` - Penalty pipeline
      * `optimizer` - Polaris optimizer
      * `opts` - Options:
        * `:log_penalties` - Whether to log penalty metrics. Default: `true`

    ## Example

        pipeline = NxPenalties.pipeline([
          {:l2, weight: 0.01}
        ])

        loop = NxPenalties.Integration.Axon.trainer(
          model,
          &Axon.Losses.mean_squared_error/2,
          pipeline,
          Polaris.Optimizers.adam(learning_rate: 0.001)
        )

        Axon.Loop.run(loop, data, %{}, epochs: 10)
    """
    @spec trainer(Axon.t(), function(), NxPenalties.Pipeline.t(), term(), keyword()) ::
            struct()
    def trainer(model, base_loss_fn, pipeline, optimizer, opts \\ []) do
      log_penalties? = Keyword.get(opts, :log_penalties, true)

      # Create regularized loss function
      loss_fn = wrap_loss_with_pipeline(base_loss_fn, pipeline)

      # Build the training loop
      loop = Axon.Loop.trainer(model, loss_fn, optimizer)

      # Optionally add penalty logging
      if log_penalties? do
        log_penalties(loop, pipeline)
      else
        loop
      end
    end

    @doc """
    Flatten nested parameter maps to a list of tensors.

    Useful for computing aggregate penalties over all model weights.

    ## Example

        params = Axon.ModelState.trainable_parameters(model_state)
        all_weights = flatten_params(params)
        total_l2 = Enum.reduce(all_weights, 0.0, fn w, acc ->
          acc + Nx.to_number(NxPenalties.l2(w))
        end)
    """
    @spec flatten_params(map() | Nx.Tensor.t()) :: [Nx.Tensor.t()]
    def flatten_params(%Nx.Tensor{} = tensor), do: [Nx.flatten(tensor)]

    def flatten_params(params) when is_map(params) do
      Enum.flat_map(params, fn {_key, value} ->
        flatten_params(value)
      end)
    end

    def flatten_params(_), do: []
  end
else
  defmodule NxPenalties.Integration.Axon do
    @moduledoc """
    Axon integration is unavailable because Axon is not installed.
    Add `{:axon, \"~> 0.6\"}` to your dependencies to enable these helpers.
    """

    @spec wrap_loss(function(), function(), keyword()) :: no_return()
    def wrap_loss(_, _, _ \\ []), do: raise_missing!()

    @spec wrap_loss_with_pipeline(function(), term(), keyword()) :: no_return()
    def wrap_loss_with_pipeline(_, _, _ \\ []), do: raise_missing!()

    @spec wrap_loss_with_params(function(), function(), keyword()) :: no_return()
    def wrap_loss_with_params(_, _, _ \\ []), do: raise_missing!()

    @spec build_train_step(term(), function(), term(), term()) :: no_return()
    def build_train_step(_, _, _, _), do: raise_missing!()

    @spec build_train_step_with_weight_decay(term(), function(), function(), term(), keyword()) ::
            no_return()
    def build_train_step_with_weight_decay(_, _, _, _, _ \\ []), do: raise_missing!()

    @spec capture_activation(term(), atom()) :: no_return()
    def capture_activation(_, _), do: raise_missing!()

    @spec extract_captures(map()) :: no_return()
    def extract_captures(_), do: raise_missing!()

    @spec build_activity_train_step(term(), function(), map(), term()) :: no_return()
    def build_activity_train_step(_, _, _, _), do: raise_missing!()

    @spec weight_schedule(map()) :: no_return()
    def weight_schedule(_), do: raise_missing!()

    @spec apply_scheduled_weights(term(), map()) :: no_return()
    def apply_scheduled_weights(_, _), do: raise_missing!()

    @spec log_penalties(struct(), term(), keyword()) :: no_return()
    def log_penalties(_, _, _ \\ []), do: raise_missing!()

    @spec schedule_weights(struct(), atom(), function(), keyword()) :: no_return()
    def schedule_weights(_, _, _, _ \\ []), do: raise_missing!()

    @spec trainer(term(), function(), term(), term(), keyword()) :: no_return()
    def trainer(_, _, _, _, _ \\ []), do: raise_missing!()

    @spec flatten_params(map() | Nx.Tensor.t()) :: no_return()
    def flatten_params(_), do: raise_missing!()

    defp raise_missing! do
      raise ArgumentError,
            "Axon integration is not available. Add {:axon, \"~> 0.6\"} to your dependencies to enable NxPenalties.Integration.Axon."
    end
  end
end
