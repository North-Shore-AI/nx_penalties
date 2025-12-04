# Polaris Integration Examples
#
# Run with: mix run examples/polaris_integration.exs
#
# This example demonstrates gradient-level weight decay transforms
# that compose with Polaris optimizers.

IO.puts("=== NxPenalties Polaris Integration ===\n")

alias NxPenalties.Integration.Polaris, as: PolarisIntegration

# Helper for nested maps (must be defined before use)
defmodule DeepMap do
  def map(%Nx.Tensor{} = tensor, fun), do: fun.(tensor)
  def map(m, fun) when is_map(m), do: Map.new(m, fn {k, v} -> {k, map(v, fun)} end)
end

# Simple SGD optimizer for demonstration
# In real usage, you'd use Polaris.Optimizers.sgd/adam/etc.
sgd_optimizer = fn learning_rate ->
  init_fn = fn _params -> %{} end

  update_fn = fn gradients, state, _params ->
    updates =
      DeepMap.map(gradients, fn g ->
        Nx.multiply(g, -learning_rate)
      end)

    {updates, state}
  end

  {init_fn, update_fn}
end

IO.puts("--- L2 Weight Decay ---")
IO.puts("L2 decay adds λ*w to gradients (decoupled weight decay)")
IO.puts("")

# Create optimizer with L2 decay
base_opt = sgd_optimizer.(0.1)
opt_with_l2 = PolarisIntegration.add_l2_decay(base_opt, 0.1)

{init_fn, update_fn} = opt_with_l2

params = %{w: Nx.tensor([2.0, -1.0])}
# Zero gradients to see pure decay
gradients = %{w: Nx.tensor([0.0, 0.0])}

state = init_fn.(params)
{updates, _state} = update_fn.(gradients, state, params)

IO.puts("Params: #{inspect(Nx.to_flat_list(params.w))}")
IO.puts("Gradients (zero): #{inspect(Nx.to_flat_list(gradients.w))}")
IO.puts("Updates with L2 decay (λ=0.1): #{inspect(Nx.to_flat_list(updates.w))}")
IO.puts("Expected: -lr * (grad + λ*w) = -0.1 * (0 + 0.1*[2,-1]) = [-0.02, 0.01]")
IO.puts("")

IO.puts("--- L1 Weight Decay ---")
IO.puts("L1 decay adds λ*sign(w) to gradients (encourages sparsity)")
IO.puts("")

opt_with_l1 = PolarisIntegration.add_l1_decay(base_opt, 0.1)
{init_fn, update_fn} = opt_with_l1

state = init_fn.(params)
{updates, _state} = update_fn.(gradients, state, params)

IO.puts("Params: #{inspect(Nx.to_flat_list(params.w))}")
IO.puts("Updates with L1 decay (λ=0.1): #{inspect(Nx.to_flat_list(updates.w))}")
IO.puts("Expected: -lr * (grad + λ*sign(w)) = -0.1 * (0 + 0.1*[1,-1]) = [-0.01, 0.01]")
IO.puts("")

IO.puts("--- Elastic Net Decay ---")
IO.puts("Combines L1 and L2: λ*(α*sign(w) + (1-α)*w)")
IO.puts("")

opt_elastic = PolarisIntegration.add_elastic_net_decay(base_opt, 0.2, 0.5)
{init_fn, update_fn} = opt_elastic

state = init_fn.(params)
{updates, _state} = update_fn.(gradients, state, params)

IO.puts("Params: #{inspect(Nx.to_flat_list(params.w))}")
IO.puts("Updates with Elastic Net (λ=0.2, α=0.5): #{inspect(Nx.to_flat_list(updates.w))}")
IO.puts("L1 term: 0.5 * sign([2,-1]) = [0.5, -0.5]")
IO.puts("L2 term: 0.5 * [2,-1] = [1.0, -0.5]")
IO.puts("Combined: 0.2 * ([0.5,-0.5] + [1.0,-0.5]) = [0.3, -0.2]")
IO.puts("Update: -0.1 * [0.3, -0.2] = [-0.03, 0.02]")
IO.puts("")

IO.puts("--- Composing Transforms ---")
IO.puts("Transforms can be piped together")
IO.puts("")

composed =
  base_opt
  |> PolarisIntegration.add_l2_decay(0.01)
  |> PolarisIntegration.add_l1_decay(0.001)

{init_fn, update_fn} = composed
state = init_fn.(params)
{updates, _state} = update_fn.(gradients, state, params)

IO.puts("With L2(0.01) + L1(0.001) decay:")
IO.puts("Updates: #{inspect(Nx.to_flat_list(updates.w))}")
IO.puts("")

IO.puts("--- Real-World Usage ---")

IO.puts("""
In practice with Polaris:

    optimizer =
      Polaris.Optimizers.adam(learning_rate: 0.001)
      |> NxPenalties.Integration.Polaris.add_l2_decay(0.01)

    # Then use in Axon training loop
    Axon.Loop.trainer(model, loss, optimizer)
""")

IO.puts("\n=== Done ===")
