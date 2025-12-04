defmodule NxPenalties.Integration.PolarisTest do
  use ExUnit.Case, async: true

  import NxPenalties.TestHelpers

  alias NxPenalties.Integration.Polaris, as: PolarisIntegration

  describe "add_l2_decay/2" do
    test "returns a valid optimizer tuple" do
      base_optimizer = sgd_optimizer(0.1)
      result = PolarisIntegration.add_l2_decay(base_optimizer, 0.01)

      assert {init_fn, update_fn} = result
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
    end

    test "modifies gradients with L2 decay term" do
      base_optimizer = sgd_optimizer(0.1)
      optimizer = PolarisIntegration.add_l2_decay(base_optimizer, 0.1)

      {init_fn, update_fn} = optimizer

      params = %{w: Nx.tensor([1.0, 2.0])}
      gradients = %{w: Nx.tensor([0.1, 0.1])}

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # Gradient should be: 0.1 + 0.1 * [1, 2] = [0.2, 0.3]
      # Update with lr=0.1: -0.1 * [0.2, 0.3] = [-0.02, -0.03]
      assert_close(updates.w, Nx.tensor([-0.02, -0.03]))
    end

    test "uses default decay value" do
      base_optimizer = sgd_optimizer(0.1)
      optimizer = PolarisIntegration.add_l2_decay(base_optimizer)

      {init_fn, update_fn} = optimizer

      params = %{w: Nx.tensor([1.0])}
      gradients = %{w: Nx.tensor([0.0])}

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # Default decay is 0.01
      # Gradient: 0 + 0.01 * 1 = 0.01
      # Update: -0.1 * 0.01 = -0.001
      assert_close(updates.w, Nx.tensor([-0.001]))
    end

    test "handles nested parameter maps" do
      base_optimizer = sgd_optimizer(0.1)
      optimizer = PolarisIntegration.add_l2_decay(base_optimizer, 0.1)

      {init_fn, update_fn} = optimizer

      params = %{
        layer1: %{w: Nx.tensor([1.0, 2.0])},
        layer2: %{w: Nx.tensor([3.0])}
      }

      gradients = %{
        layer1: %{w: Nx.tensor([0.0, 0.0])},
        layer2: %{w: Nx.tensor([0.0])}
      }

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # Decay only (gradient = 0)
      # layer1.w: -0.1 * (0.1 * [1, 2]) = [-0.01, -0.02]
      # layer2.w: -0.1 * (0.1 * [3]) = [-0.03]
      assert_close(updates.layer1.w, Nx.tensor([-0.01, -0.02]))
      assert_close(updates.layer2.w, Nx.tensor([-0.03]))
    end
  end

  describe "add_l1_decay/2" do
    test "returns a valid optimizer tuple" do
      base_optimizer = sgd_optimizer(0.1)
      result = PolarisIntegration.add_l1_decay(base_optimizer, 0.001)

      assert {init_fn, update_fn} = result
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
    end

    test "uses sign of weights for L1 decay" do
      base_optimizer = sgd_optimizer(0.1)
      optimizer = PolarisIntegration.add_l1_decay(base_optimizer, 0.1)

      {init_fn, update_fn} = optimizer

      params = %{w: Nx.tensor([2.0, -3.0])}
      gradients = %{w: Nx.tensor([0.0, 0.0])}

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # Gradient: 0 + 0.1 * sign([2, -3]) = [0.1, -0.1]
      # Update: -0.1 * [0.1, -0.1] = [-0.01, 0.01]
      assert_close(updates.w, Nx.tensor([-0.01, 0.01]))
    end

    test "handles zero weights correctly" do
      base_optimizer = sgd_optimizer(0.1)
      optimizer = PolarisIntegration.add_l1_decay(base_optimizer, 0.1)

      {init_fn, update_fn} = optimizer

      params = %{w: Nx.tensor([0.0, 1.0])}
      gradients = %{w: Nx.tensor([0.0, 0.0])}

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # sign(0) = 0 in Nx
      # Gradient: [0 + 0.1*0, 0 + 0.1*1] = [0, 0.1]
      # Update: [-0, -0.01] = [0, -0.01]
      assert_close(updates.w, Nx.tensor([0.0, -0.01]))
    end
  end

  describe "add_elastic_net_decay/3" do
    test "returns a valid optimizer tuple" do
      base_optimizer = sgd_optimizer(0.1)
      result = PolarisIntegration.add_elastic_net_decay(base_optimizer, 0.01, 0.5)

      assert {init_fn, update_fn} = result
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
    end

    test "combines L1 and L2 decay" do
      base_optimizer = sgd_optimizer(1.0)
      # 50% L1, 50% L2
      optimizer = PolarisIntegration.add_elastic_net_decay(base_optimizer, 1.0, 0.5)

      {init_fn, update_fn} = optimizer

      params = %{w: Nx.tensor([2.0])}
      gradients = %{w: Nx.tensor([0.0])}

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # L1 term: 0.5 * sign(2) = 0.5
      # L2 term: 0.5 * 2 = 1.0
      # Total decay: 1.0 * (0.5 + 1.0) = 1.5
      # Update: -1.0 * 1.5 = -1.5
      assert_close(updates.w, Nx.tensor([-1.5]))
    end

    test "pure L1 with l1_ratio=1.0" do
      base_optimizer = sgd_optimizer(1.0)
      optimizer = PolarisIntegration.add_elastic_net_decay(base_optimizer, 0.1, 1.0)

      {init_fn, update_fn} = optimizer

      params = %{w: Nx.tensor([2.0])}
      gradients = %{w: Nx.tensor([0.0])}

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # L1 only: 0.1 * sign(2) * 1.0 + 0.1 * 2 * 0.0 = 0.1
      # Update: -1.0 * 0.1 = -0.1
      assert_close(updates.w, Nx.tensor([-0.1]))
    end

    test "pure L2 with l1_ratio=0.0" do
      base_optimizer = sgd_optimizer(1.0)
      optimizer = PolarisIntegration.add_elastic_net_decay(base_optimizer, 0.1, 0.0)

      {init_fn, update_fn} = optimizer

      params = %{w: Nx.tensor([2.0])}
      gradients = %{w: Nx.tensor([0.0])}

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # L2 only: 0.1 * 2 = 0.2
      # Update: -1.0 * 0.2 = -0.2
      assert_close(updates.w, Nx.tensor([-0.2]))
    end
  end

  describe "optimizer composition" do
    test "transforms compose via piping" do
      optimizer =
        sgd_optimizer(0.1)
        |> PolarisIntegration.add_l2_decay(0.01)
        |> PolarisIntegration.add_l1_decay(0.001)

      {init_fn, update_fn} = optimizer

      params = %{w: Nx.tensor([1.0])}
      gradients = %{w: Nx.tensor([0.0])}

      state = init_fn.(params)
      {updates, _new_state} = update_fn.(gradients, state, params)

      # Both decays should be applied
      # L2 decay: 0.01 * 1.0 = 0.01
      # L1 decay: 0.001 * sign(1.0) = 0.001
      # Total gradient: 0 + 0.01 + 0.001 = 0.011
      # Update: -0.1 * 0.011 = -0.0011
      assert_close(updates.w, Nx.tensor([-0.0011]), atol: 1.0e-5)
    end

    test "state is properly maintained across updates" do
      optimizer = PolarisIntegration.add_l2_decay(sgd_optimizer(0.1), 0.1)

      {init_fn, update_fn} = optimizer

      params = %{w: Nx.tensor([1.0])}
      gradients = %{w: Nx.tensor([0.1])}

      state = init_fn.(params)
      {_updates1, state1} = update_fn.(gradients, state, params)
      {_updates2, state2} = update_fn.(gradients, state1, params)

      # State should be updated (contains base optimizer state)
      assert is_map(state2)
    end
  end

  # Helper to create a simple SGD optimizer for testing
  defp sgd_optimizer(learning_rate) do
    init_fn = fn _params -> %{} end

    update_fn = fn gradients, state, _params ->
      updates =
        deep_map_single(gradients, fn g ->
          Nx.multiply(g, -learning_rate)
        end)

      {updates, state}
    end

    {init_fn, update_fn}
  end

  # Deep map over single nested structure (handles both maps and tensors)
  # Check for tensors first since Nx.Tensor is a struct (which is also a map)
  defp deep_map_single(%Nx.Tensor{} = tensor, fun) do
    fun.(tensor)
  end

  defp deep_map_single(structure, fun) when is_map(structure) do
    Map.new(structure, fn {key, value} ->
      {key, deep_map_single(value, fun)}
    end)
  end

  defp deep_map_single(other, fun) do
    fun.(other)
  end
end
