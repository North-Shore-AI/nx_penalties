# Axon Training Integration Example
#
# Run with: mix run examples/axon_training.exs
#
# Note: Requires Axon as a dependency. This example demonstrates
# how to integrate NxPenalties with Axon training loops.

IO.puts("=== NxPenalties Axon Integration ===\n")

# Check if Axon is available
case Code.ensure_loaded(Axon) do
  {:module, _} ->
    IO.puts("Axon is available - running full example\n")

    # Build a simple model
    model =
      Axon.input("input", shape: {nil, 10})
      |> Axon.dense(32, activation: :relu)
      |> Axon.dense(16, activation: :relu)
      |> Axon.dense(1)

    IO.puts("Model architecture:")
    IO.puts("  Input: {batch, 10}")
    IO.puts("  Dense: 32 units, ReLU")
    IO.puts("  Dense: 16 units, ReLU")
    IO.puts("  Output: 1 unit")
    IO.puts("")

    # Create a penalty pipeline
    pipeline =
      NxPenalties.pipeline([
        {:l2, weight: 0.01},
        {:l1, weight: 0.001}
      ])

    IO.puts("Penalty pipeline created:")
    IO.puts("  L2 weight: 0.01")
    IO.puts("  L1 weight: 0.001")
    IO.puts("")

    # Generate synthetic data
    key = Nx.Random.key(42)
    {x_train, key} = Nx.Random.uniform(key, 0.0, 1.0, shape: {100, 10}, type: :f32)
    {noise, _key} = Nx.Random.normal(key, 0.0, 0.1, shape: {100, 1}, type: :f32)

    # Target: sum of first 3 features + noise
    y_train =
      x_train
      |> Nx.slice_along_axis(0, 3, axis: 1)
      |> Nx.sum(axes: [1], keep_axes: true)
      |> Nx.add(noise)

    IO.puts("Generated synthetic training data:")
    IO.puts("  X shape: #{inspect(Nx.shape(x_train))}")
    IO.puts("  Y shape: #{inspect(Nx.shape(y_train))}")
    IO.puts("  Target: sum of first 3 features + noise")
    IO.puts("")

    # Using the Axon integration helper
    IO.puts("--- Using NxPenalties with Axon ---")

    {init_fn, predict_fn} = Axon.build(model)
    model_state = init_fn.(Nx.template({1, 10}, :f32), %{})

    # Get the parameters from model state
    params = Axon.ModelState.trainable_parameters(model_state)

    # Get weights for demonstration
    first_layer_weights = params["dense_0"]["kernel"]
    IO.puts("First layer weights shape: #{inspect(Nx.shape(first_layer_weights))}")

    # Compute regularization on weights
    {reg_loss, metrics} = NxPenalties.compute(pipeline, first_layer_weights)
    IO.puts("Regularization loss on first layer: #{Nx.to_number(reg_loss)}")
    IO.puts("Metrics: #{inspect(Map.keys(metrics))}")
    IO.puts("")

    # Training step function using NxPenalties helper
    IO.puts("--- Training Step with Regularization ---")

    # Helper to extract all kernels from params
    extract_kernels = fn params ->
      params
      |> Map.values()
      |> Enum.flat_map(fn layer ->
        case layer do
          %{"kernel" => k} -> [k]
          _ -> []
        end
      end)
    end

    step_fn = fn params, x_batch, y_batch ->
      # Simple MSE + regularization
      {loss, grads} =
        Nx.Defn.value_and_grad(params, fn p ->
          y_pred = predict_fn.(p, x_batch)
          mse = Nx.mean(Nx.pow(Nx.subtract(y_pred, y_batch), 2))

          # Add L2 penalty on all kernels
          reg =
            p
            |> extract_kernels.()
            |> Enum.map(&NxPenalties.l2(&1, lambda: 0.01))
            |> Enum.reduce(Nx.tensor(0.0), &Nx.add/2)

          Nx.add(mse, reg)
        end)

      # Simple gradient descent update
      new_params =
        params
        |> Map.keys()
        |> Enum.reduce(params, fn layer_name, acc ->
          layer_grads = grads[layer_name]
          layer_params = acc[layer_name]

          updated_layer =
            layer_params
            |> Map.keys()
            |> Enum.reduce(layer_params, fn param_name, layer_acc ->
              grad = layer_grads[param_name]
              param = layer_acc[param_name]
              # Learning rate = 0.01
              Map.put(layer_acc, param_name, Nx.subtract(param, Nx.multiply(grad, 0.01)))
            end)

          Map.put(acc, layer_name, updated_layer)
        end)

      {loss, new_params}
    end

    # Run a few training steps
    IO.puts("Running 5 training steps...")

    {final_loss, _final_params} =
      Enum.reduce(1..5, {nil, params}, fn step, {_prev_loss, p} ->
        {loss, new_p} = step_fn.(p, x_train, y_train)
        IO.puts("  Step #{step}: loss = #{Float.round(Nx.to_number(loss), 6)}")
        {loss, new_p}
      end)

    IO.puts("\nFinal loss: #{Nx.to_number(final_loss)}")

  {:error, _} ->
    IO.puts("Axon is not available - showing conceptual example\n")

    IO.puts("""
    To use NxPenalties with Axon, add Axon to your dependencies:

        {:axon, "~> 0.6"}

    Then in your training loop:

        # Create penalty pipeline
        pipeline = NxPenalties.pipeline([
          {:l2, weight: 0.01},
          {:l1, weight: 0.001}
        ])

        # Custom training step
        defn train_step(model_fn, params, x, y, pipeline) do
          {loss, grads} = value_and_grad(params, fn p ->
            pred = model_fn.(p, x)
            base_loss = Axon.Losses.mean_squared_error(pred, y)

            # Add regularization
            reg = compute_weight_penalty(p, pipeline)
            base_loss + reg
          end)

          # Update params...
        end

    See the NxPenalties.Integration.Axon module for helper functions.
    """)
end

IO.puts("\n=== Done ===")
