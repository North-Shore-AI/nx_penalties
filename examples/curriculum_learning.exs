# Curriculum Learning Example
#
# Run with: mix run examples/curriculum_learning.exs
#
# Demonstrates dynamic penalty weight adjustment for curriculum learning,
# where regularization strength changes throughout training.

IO.puts("=== NxPenalties Curriculum Learning ===\n")

# Create initial pipeline with high regularization
pipeline =
  NxPenalties.pipeline([
    # Start with strong sparsity
    {:l1, weight: 0.1},
    # Moderate weight decay
    {:l2, weight: 0.01}
  ])

IO.puts("Initial pipeline configuration:")
IO.puts("  L1 weight: 0.1 (strong sparsity)")
IO.puts("  L2 weight: 0.01 (moderate decay)")
IO.puts("")

# Sample weights tensor
key = Nx.Random.key(42)
{weights, _} = Nx.Random.normal(key, 0.0, 1.0, shape: {100, 50}, type: :f32)
IO.puts("Simulated model weights: #{inspect(Nx.shape(weights))}")
IO.puts("")

# Curriculum learning: gradually reduce regularization
IO.puts("--- Curriculum Learning Schedule ---")
IO.puts("Simulating training epochs with decreasing regularization:\n")

# Define curriculum schedule
schedule = [
  # Epoch 1: Strong regularization
  {1, 0.1, 0.01},
  # Epoch 10: Moderate
  {10, 0.05, 0.005},
  # Epoch 25: Light
  {25, 0.01, 0.001},
  # Epoch 50: Very light
  {50, 0.001, 0.0001}
]

Enum.each(schedule, fn {epoch, l1_weight, l2_weight} ->
  # Update pipeline weights
  updated_pipeline =
    pipeline
    |> NxPenalties.Pipeline.update_weight(:l1, l1_weight)
    |> NxPenalties.Pipeline.update_weight(:l2, l2_weight)

  # Compute penalty
  {total, metrics} = NxPenalties.compute(updated_pipeline, weights)

  IO.puts("Epoch #{epoch}:")
  IO.puts("  L1 weight: #{l1_weight}, L2 weight: #{l2_weight}")
  IO.puts("  L1 penalty: #{Float.round(metrics["l1_weighted"], 4)}")
  IO.puts("  L2 penalty: #{Float.round(metrics["l2_weighted"], 4)}")
  IO.puts("  Total: #{Float.round(Nx.to_number(total), 4)}")
  IO.puts("")
end)

# Alternative: Enable/disable penalties during training
IO.puts("--- Phase-Based Training ---")
IO.puts("Some training regimes enable different penalties in phases:\n")

# Phase 1: Only L2 (prevent explosion during warmup)
phase1 =
  pipeline
  |> NxPenalties.Pipeline.set_enabled(:l1, false)
  |> NxPenalties.Pipeline.set_enabled(:l2, true)

{p1_total, _} = NxPenalties.compute(phase1, weights)
IO.puts("Phase 1 (warmup - L2 only):")
IO.puts("  Active penalties: L2")
IO.puts("  Total penalty: #{Float.round(Nx.to_number(p1_total), 4)}")
IO.puts("")

# Phase 2: Both penalties (main training)
phase2 =
  pipeline
  |> NxPenalties.Pipeline.set_enabled(:l1, true)
  |> NxPenalties.Pipeline.set_enabled(:l2, true)

{p2_total, _} = NxPenalties.compute(phase2, weights)
IO.puts("Phase 2 (main training - L1 + L2):")
IO.puts("  Active penalties: L1, L2")
IO.puts("  Total penalty: #{Float.round(Nx.to_number(p2_total), 4)}")
IO.puts("")

# Phase 3: L1 only (encourage sparsity for pruning)
phase3 =
  pipeline
  |> NxPenalties.Pipeline.update_weight(:l1, 0.5)
  |> NxPenalties.Pipeline.set_enabled(:l1, true)
  |> NxPenalties.Pipeline.set_enabled(:l2, false)

{p3_total, _} = NxPenalties.compute(phase3, weights)
IO.puts("Phase 3 (pruning prep - L1 only, high weight):")
IO.puts("  Active penalties: L1 (weight: 0.5)")
IO.puts("  Total penalty: #{Float.round(Nx.to_number(p3_total), 4)}")
IO.puts("")

# Using Elastic Net with changing ratio
IO.puts("--- Elastic Net Curriculum ---")
IO.puts("Elastic Net ratio can shift from L2-heavy to L1-heavy:\n")

ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

Enum.each(ratios, fn ratio ->
  # Need to rebuild pipeline with new ratio
  updated_elastic =
    NxPenalties.pipeline([
      {:elastic_net, weight: 0.01, l1_ratio: ratio}
    ])

  {total, metrics} = NxPenalties.compute(updated_elastic, weights)
  IO.puts("L1 ratio: #{ratio}")
  IO.puts("  Raw penalty: #{Float.round(metrics["elastic_net"], 4)}")
  IO.puts("  Total (weighted): #{Float.round(Nx.to_number(total), 4)}")
end)

IO.puts("")

# Gradient flow verification
IO.puts("--- Gradient Compatibility Check ---")
IO.puts("Verifying gradients flow correctly through curriculum pipeline:\n")

# Create a pipeline for gradient testing
grad_pipeline =
  NxPenalties.pipeline([
    {:l2, weight: 0.01}
  ])

# Compute gradients
grad_fn = fn t ->
  NxPenalties.compute_total(grad_pipeline, t)
end

small_weights = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
grads = Nx.Defn.grad(grad_fn).(small_weights)

IO.puts("Input weights:")
IO.puts("  #{inspect(Nx.to_flat_list(small_weights))}")
IO.puts("")
IO.puts("L2 gradients (∂L2/∂w = 2λw):")
IO.puts("  #{inspect(Nx.to_flat_list(grads))}")
IO.puts("")

# Verify gradient correctness
expected_grads = Nx.multiply(small_weights, 2.0 * 0.01)
max_diff = Nx.reduce_max(Nx.abs(Nx.subtract(grads, expected_grads))) |> Nx.to_number()
IO.puts("Max difference from expected: #{max_diff}")
IO.puts("Gradients correct: #{max_diff < 1.0e-5}")

IO.puts("\n=== Done ===")
