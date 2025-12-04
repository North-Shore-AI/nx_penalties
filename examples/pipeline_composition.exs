# Pipeline Composition Examples
#
# Run with: mix run examples/pipeline_composition.exs

IO.puts("=== NxPenalties Pipeline Composition ===\n")

# Build a pipeline with multiple penalties
pipeline =
  NxPenalties.pipeline([
    {:l1, weight: 0.001},
    {:l2, weight: 0.01}
  ])

IO.puts("Created pipeline with L1 and L2 penalties")
IO.puts("")

# Sample tensor (e.g., model outputs or weights)
tensor = Nx.tensor([1.0, -2.0, 3.0, -0.5, 2.0])
IO.puts("Input tensor: #{inspect(Nx.to_flat_list(tensor))}")

# Compute penalties
{total, metrics} = NxPenalties.compute(pipeline, tensor)

IO.puts("\n--- Pipeline Results ---")
IO.puts("Total penalty: #{Nx.to_number(total)}")
IO.puts("\nMetrics:")

Enum.each(metrics, fn {key, value} ->
  IO.puts("  #{key}: #{value}")
end)

IO.puts("")

# Dynamic weight adjustment (useful for curriculum learning)
IO.puts("--- Dynamic Weight Adjustment ---")

updated_pipeline =
  pipeline
  |> NxPenalties.Pipeline.update_weight(:l1, 0.01)
  |> NxPenalties.Pipeline.update_weight(:l2, 0.001)

{new_total, new_metrics} = NxPenalties.compute(updated_pipeline, tensor)
IO.puts("After swapping weights:")
IO.puts("  New total: #{Nx.to_number(new_total)}")
IO.puts("  L1 weighted: #{new_metrics["l1_weighted"]}")
IO.puts("  L2 weighted: #{new_metrics["l2_weighted"]}")

IO.puts("")

# Enable/disable penalties
IO.puts("--- Enable/Disable Penalties ---")

disabled_pipeline = NxPenalties.Pipeline.set_enabled(pipeline, :l2, false)
{disabled_total, disabled_metrics} = NxPenalties.compute(disabled_pipeline, tensor)
IO.puts("With L2 disabled:")
IO.puts("  Total: #{Nx.to_number(disabled_total)}")
IO.puts("  Metrics: #{inspect(Map.keys(disabled_metrics))}")

IO.puts("")

# Using compute_total for gradient-compatible computation
IO.puts("--- Gradient-Compatible Computation ---")
total_only = NxPenalties.compute_total(pipeline, tensor)
IO.puts("Total (no metrics): #{Nx.to_number(total_only)}")

# Demonstrate gradient flow
grad_fn = Nx.Defn.grad(fn t -> NxPenalties.compute_total(pipeline, t) end)
grads = grad_fn.(tensor)
IO.puts("Gradients: #{inspect(Nx.to_flat_list(grads))}")

IO.puts("\n=== Done ===")
