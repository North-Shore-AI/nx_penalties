# Gradient Penalty Examples
# Run with: mix run examples/gradient_penalty.exs

alias NxPenalties.GradientPenalty

IO.puts("=== Gradient Penalty Examples ===\n")

# Basic gradient penalty
IO.puts("1. Basic Gradient Penalty")
IO.puts("-" |> String.duplicate(40))

loss_fn = fn x -> Nx.sum(x) end
tensor = Nx.tensor([1.0, 2.0, 3.0, 4.0])

# Gradient of sum is all ones: [1,1,1,1], norm = 2
penalty = GradientPenalty.gradient_penalty(loss_fn, tensor, target_norm: 1.0)
IO.puts("Loss function: sum(x)")
IO.puts("Tensor: [1, 2, 3, 4]")
IO.puts("Gradient: [1, 1, 1, 1] (norm = 2)")
IO.puts("Target norm: 1.0")
IO.puts("Penalty: #{Nx.to_number(penalty)}")
IO.puts("Expected: (2 - 1)² = 1.0")

# Zero penalty when target matches
penalty_zero = GradientPenalty.gradient_penalty(loss_fn, tensor, target_norm: 2.0)
IO.puts("\nWith target_norm: 2.0, penalty: #{Nx.to_number(penalty_zero)}")
IO.puts("Expected: ~0.0")

# Output magnitude penalty (cheaper proxy)
IO.puts("\n2. Output Magnitude Penalty (Cheap Proxy)")
IO.puts("-" |> String.duplicate(40))

output = Nx.tensor([3.0, 4.0])
# Magnitude = 5 (L2 norm)
mag_penalty = GradientPenalty.output_magnitude_penalty(output, target: 5.0, reduction: :sum)
IO.puts("Output: [3, 4], L2 norm = 5")
IO.puts("Target: 5.0")
IO.puts("Penalty: #{Nx.to_number(mag_penalty)}")
IO.puts("Expected: ~0.0")

mag_penalty_off = GradientPenalty.output_magnitude_penalty(output, target: 1.0, reduction: :sum)
IO.puts("\nWith target: 1.0, penalty: #{Nx.to_number(mag_penalty_off)}")
IO.puts("Expected: (5 - 1)² = 16.0")

# Interpolated gradient penalty (WGAN-GP style)
IO.puts("\n3. Interpolated Gradient Penalty (WGAN-GP Style)")
IO.puts("-" |> String.duplicate(40))

loss_fn_l2 = fn x -> Nx.sum(Nx.pow(x, 2)) end
real = Nx.tensor([1.0, 0.0, 0.0])
fake = Nx.tensor([0.0, 1.0, 0.0])

interp_penalty =
  GradientPenalty.interpolated_gradient_penalty(
    loss_fn_l2,
    fake,
    real,
    target_norm: 1.0
  )

IO.puts("Loss function: sum(x²)")
IO.puts("Real: [1, 0, 0], Fake: [0, 1, 0]")
IO.puts("Interpolated penalty: #{Nx.to_number(interp_penalty)}")

# Performance comparison
IO.puts("\n4. Performance Comparison")
IO.puts("-" |> String.duplicate(40))

{large_tensor, _} = Nx.Random.uniform(Nx.Random.key(42), shape: {100, 100})

{time_grad, _} =
  :timer.tc(fn ->
    GradientPenalty.gradient_penalty(loss_fn, large_tensor)
  end)

{time_mag, _} =
  :timer.tc(fn ->
    GradientPenalty.output_magnitude_penalty(large_tensor)
  end)

IO.puts("Tensor size: 100x100 = 10,000 elements")
IO.puts("Gradient penalty time: #{time_grad} µs")
IO.puts("Output magnitude time: #{time_mag} µs")
IO.puts("Speedup: #{Float.round(time_grad / max(time_mag, 1), 1)}x")
IO.puts("\nRecommendation: Use output_magnitude_penalty for frequent calls")

# Practical usage pattern
IO.puts("\n5. Practical Usage Pattern")
IO.puts("-" |> String.duplicate(40))

IO.puts("""
# Apply gradient penalty every N steps to reduce cost:

def train_step(model, batch, step) do
  # ... forward pass ...

  loss = base_loss

  # Add gradient penalty every 10 steps
  loss = if rem(step, 10) == 0 do
    gp = GradientPenalty.gradient_penalty(model_fn, inputs, target_norm: 1.0)
    Nx.add(loss, Nx.multiply(gp, 10.0))  # WGAN-GP uses weight 10
  else
    loss
  end

  # ... backward pass ...
end
""")
