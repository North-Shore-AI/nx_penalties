# Constraints Examples (v0.2)
#
# Run with: mix run examples/constraints.exs
#
# This example demonstrates structural constraint penalties:
# - Orthogonality: decorrelate representations
# - Consistency: paired output stability

IO.puts("=== NxPenalties Constraints ===\n")

alias NxPenalties.Constraints

# ============================================================================
# Orthogonality Penalty
# ============================================================================

IO.puts("--- Orthogonality Penalty ---")
IO.puts("Encourages uncorrelated rows in weight matrices")
IO.puts("")

# Identity matrix has orthogonal rows
identity = Nx.eye(3)
IO.puts("Identity matrix (orthogonal rows):")
IO.inspect(identity, label: "Matrix")

penalty = Constraints.orthogonality(identity, mode: :soft)
IO.puts("Orthogonality penalty (soft): #{Nx.to_number(penalty)}")
IO.puts("Expected: ~0 (rows are already orthogonal)")
IO.puts("")

# Correlated rows (all same direction)
correlated =
  Nx.tensor([
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0]
  ])

IO.puts("Correlated matrix (identical rows):")
IO.inspect(correlated, label: "Matrix")

penalty = Constraints.orthogonality(correlated, mode: :soft)
IO.puts("Orthogonality penalty (soft): #{Nx.to_number(penalty)}")
IO.puts("Expected: high value (rows are maximally correlated)")
IO.puts("")

# Soft vs Hard mode
IO.puts("--- Soft vs Hard Mode ---")

slightly_off =
  Nx.tensor([
    [1.0, 0.1],
    [0.1, 1.0]
  ])

soft = Constraints.orthogonality(slightly_off, mode: :soft)
hard = Constraints.orthogonality(slightly_off, mode: :hard)
IO.puts("Slightly non-orthogonal matrix:")
IO.inspect(slightly_off, label: "Matrix")
IO.puts("Soft penalty (off-diagonal only): #{Nx.to_number(soft)}")
IO.puts("Hard penalty (deviation from I): #{Nx.to_number(hard)}")
IO.puts("")

# ============================================================================
# Consistency Penalty
# ============================================================================

IO.puts("--- Consistency Penalty ---")
IO.puts("Penalizes divergence between paired outputs")
IO.puts("Use case: clean vs augmented input predictions should match")
IO.puts("")

# Identical outputs = zero penalty
clean_output = Nx.tensor([1.0, 2.0, 3.0])
augmented_output = Nx.tensor([1.0, 2.0, 3.0])

penalty = Constraints.consistency(clean_output, augmented_output)
IO.puts("Identical outputs:")
IO.puts("  Clean: #{inspect(Nx.to_flat_list(clean_output))}")
IO.puts("  Augmented: #{inspect(Nx.to_flat_list(augmented_output))}")
IO.puts("  Consistency penalty (MSE): #{Nx.to_number(penalty)}")
IO.puts("")

# Different outputs
augmented_noisy = Nx.tensor([1.1, 2.2, 2.8])
penalty = Constraints.consistency(clean_output, augmented_noisy)
IO.puts("Slightly different outputs:")
IO.puts("  Clean: #{inspect(Nx.to_flat_list(clean_output))}")
IO.puts("  Augmented: #{inspect(Nx.to_flat_list(augmented_noisy))}")
IO.puts("  Consistency penalty (MSE): #{Nx.to_number(penalty)}")
IO.puts("")

# Different metrics
IO.puts("--- Different Metrics ---")
o1 = Nx.tensor([1.0, 2.0, 3.0])
o2 = Nx.tensor([2.0, 3.0, 4.0])

mse = Constraints.consistency(o1, o2, metric: :mse)
l1 = Constraints.consistency(o1, o2, metric: :l1)
cosine = Constraints.consistency(o1, o2, metric: :cosine)

IO.puts("Outputs differ by [1, 1, 1]:")
IO.puts("  MSE metric: #{Nx.to_number(mse)}")
IO.puts("  L1 metric: #{Nx.to_number(l1)}")
IO.puts("  Cosine metric: #{Nx.to_number(cosine)}")
IO.puts("")

# Cosine distance examples
IO.puts("--- Cosine Distance ---")
same_dir = Constraints.consistency(Nx.tensor([1.0, 0.0]), Nx.tensor([2.0, 0.0]), metric: :cosine)

orthogonal =
  Constraints.consistency(Nx.tensor([1.0, 0.0]), Nx.tensor([0.0, 1.0]), metric: :cosine)

opposite = Constraints.consistency(Nx.tensor([1.0, 0.0]), Nx.tensor([-1.0, 0.0]), metric: :cosine)

IO.puts("Same direction: #{Nx.to_number(same_dir)} (expected: 0)")
IO.puts("Orthogonal: #{Nx.to_number(orthogonal)} (expected: 1)")
IO.puts("Opposite direction: #{Nx.to_number(opposite)} (expected: 2)")
IO.puts("")

# ============================================================================
# Using Constraints in Pipeline
# ============================================================================

IO.puts("--- Using in Pipeline ---")

IO.puts("""
Constraints can be added to pipelines:

    # For orthogonality on hidden states
    pipeline =
      NxPenalties.Pipeline.new()
      |> NxPenalties.Pipeline.add(
           :ortho,
           fn tensor, _opts -> NxPenalties.Constraints.orthogonality(tensor) end,
           weight: 0.01
         )

    # For consistency (needs two tensors)
    consistency_loss = fn clean, noisy ->
      NxPenalties.Constraints.consistency(clean, noisy, metric: :mse)
    end
""")

IO.puts("\n=== Done ===")
