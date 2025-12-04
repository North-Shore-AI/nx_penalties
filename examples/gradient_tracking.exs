# Gradient Tracking Examples
#
# Run with: mix run examples/gradient_tracking.exs
#
# This example demonstrates monitoring gradient norms from regularization
# penalties - useful for debugging training dynamics.

IO.puts("=== NxPenalties Gradient Tracking ===\n")

# Create a pipeline with multiple penalties
pipeline =
  NxPenalties.pipeline([
    {:l1, weight: 0.001},
    {:l2, weight: 0.01}
  ])

tensor = Nx.tensor([1.0, 2.0, 3.0, 4.0])
IO.puts("Input tensor: #{inspect(Nx.to_flat_list(tensor))}")
IO.puts("")

# Standard compute (no gradient tracking)
IO.puts("--- Standard Computation ---")
{total, metrics} = NxPenalties.compute(pipeline, tensor)
IO.puts("Total penalty: #{Nx.to_number(total)}")
IO.puts("Metrics keys: #{inspect(Map.keys(metrics))}")
IO.puts("")

# With gradient tracking enabled
IO.puts("--- With Gradient Tracking ---")
{total, metrics} = NxPenalties.compute(pipeline, tensor, track_grad_norms: true)
IO.puts("Total penalty: #{Nx.to_number(total)}")
IO.puts("")
IO.puts("Gradient norms:")
IO.puts("  L1 grad norm: #{metrics["l1_grad_norm"]}")
IO.puts("  L2 grad norm: #{metrics["l2_grad_norm"]}")
IO.puts("  Total grad norm: #{metrics["total_grad_norm"]}")
IO.puts("")

# Note: Per-penalty norms are for the unweighted penalties.
# Total norm is for the weighted sum. They don't sum directly.
IO.puts("Note: L1/L2 norms are unweighted. Total norm includes weights.")
IO.puts("L2 has higher gradient norm but lower weight (0.01 vs 0.001)")
IO.puts("")

# Using GradientTracker directly for custom loss functions
IO.puts("--- Direct GradientTracker Usage ---")
alias NxPenalties.GradientTracker

# Custom loss function
custom_loss = fn x -> Nx.sum(Nx.pow(x, 3)) end
norm = GradientTracker.compute_grad_norm(custom_loss, tensor)
IO.puts("Gradient norm of sum(x^3): #{norm}")
# Gradient of x^3 is 3x^2, so for [1,2,3,4] -> [3,12,27,48]
# L2 norm = sqrt(9 + 144 + 729 + 2304) = sqrt(3186)
IO.puts("Expected: sqrt(3186) = #{:math.sqrt(3186)}")
IO.puts("")

# Validation helper
IO.puts("--- Tensor Validation ---")
valid_tensor = Nx.tensor([1.0, 2.0, 3.0])

case NxPenalties.validate(valid_tensor) do
  {:ok, _} -> IO.puts("Valid tensor: OK")
  {:error, reason} -> IO.puts("Valid tensor: Error - #{reason}")
end

nan_tensor = Nx.Constants.nan({:f, 32})

case NxPenalties.validate(nan_tensor) do
  {:ok, _} -> IO.puts("NaN tensor: OK")
  {:error, reason} -> IO.puts("NaN tensor: Error - #{reason}")
end

inf_tensor = Nx.Constants.infinity({:f, 32})

case NxPenalties.validate(inf_tensor) do
  {:ok, _} -> IO.puts("Inf tensor: OK")
  {:error, reason} -> IO.puts("Inf tensor: Error - #{reason}")
end

IO.puts("\n=== Done ===")
