# Constraints Examples
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

# Spectral Orthogonality
IO.puts("--- Spectral Orthogonality ---")

# Identity matrix - perfectly orthonormal
identity = Nx.eye(4)
spectral_identity = Constraints.orthogonality(identity, mode: :spectral)
IO.puts("Spectral penalty (identity): #{Nx.to_number(spectral_identity)}")
IO.puts("Expected: ~0.0")
IO.puts("")

# Diagonal matrix with different scales
diagonal =
  Nx.tensor([
    [2.0, 0.0, 0.0],
    [0.0, 3.0, 0.0],
    [0.0, 0.0, 1.0]
  ])

spectral_diag_norm = Constraints.orthogonality(diagonal, mode: :spectral, normalize: true)
spectral_diag_raw = Constraints.orthogonality(diagonal, mode: :spectral, normalize: false)
IO.puts("Diagonal matrix with different scales:")
IO.inspect(diagonal, label: "Matrix")
IO.puts("Spectral penalty (normalized): #{Nx.to_number(spectral_diag_norm)}")
IO.puts("Spectral penalty (unnormalized): #{Nx.to_number(spectral_diag_raw)}")
IO.puts("")

# Compare all three modes
IO.puts("--- Comparing All Modes ---")
{random_matrix, _key} = Nx.Random.uniform(Nx.Random.key(42), shape: {4, 8})
soft = Constraints.orthogonality(random_matrix, mode: :soft)
hard = Constraints.orthogonality(random_matrix, mode: :hard)
spectral = Constraints.orthogonality(random_matrix, mode: :spectral)
IO.puts("Random matrix penalties (4x8):")
IO.puts("  Soft: #{Nx.to_number(soft)}")
IO.puts("  Hard: #{Nx.to_number(hard)}")
IO.puts("  Spectral: #{Nx.to_number(spectral)}")
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
# KL Consistency for probability distributions
# ============================================================================

IO.puts("--- KL Consistency ---")
IO.puts("Symmetric KL divergence for log-probability distributions")
IO.puts("")

# Create log-probability distributions
vocab_size = 8
uniform = Nx.broadcast(Nx.tensor(-:math.log(vocab_size)), {1, vocab_size})
peaked = Nx.tensor([[0.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0]])

# Same distribution = 0 consistency loss
same_kl = Constraints.consistency(uniform, uniform, metric: :kl)
IO.puts("KL consistency (same): #{Nx.to_number(same_kl)}")
IO.puts("Expected: 0.0")
IO.puts("")

# Different distributions = positive consistency loss
diff_kl = Constraints.consistency(uniform, peaked, metric: :kl)
IO.puts("KL consistency (uniform vs peaked): #{Nx.to_number(diff_kl)}")
IO.puts("Expected: Positive value")
IO.puts("")

# Verify symmetry
reverse_kl = Constraints.consistency(peaked, uniform, metric: :kl)
IO.puts("KL consistency (peaked vs uniform): #{Nx.to_number(reverse_kl)}")
IO.puts("Expected: Same as above (symmetric)")
IO.puts("")

# Different reduction modes
IO.puts("--- KL with different reductions ---")

batch_logprobs_p =
  Nx.tensor([
    [-1.386, -1.386, -1.386, -1.386],
    [0.0, -10.0, -10.0, -10.0]
  ])

batch_logprobs_q =
  Nx.tensor([
    [0.0, -10.0, -10.0, -10.0],
    [-1.386, -1.386, -1.386, -1.386]
  ])

kl_mean =
  Constraints.consistency(batch_logprobs_p, batch_logprobs_q, metric: :kl, reduction: :mean)

kl_sum = Constraints.consistency(batch_logprobs_p, batch_logprobs_q, metric: :kl, reduction: :sum)

kl_none =
  Constraints.consistency(batch_logprobs_p, batch_logprobs_q, metric: :kl, reduction: :none)

IO.puts("KL with reduction :mean: #{Nx.to_number(kl_mean)}")
IO.puts("KL with reduction :sum: #{Nx.to_number(kl_sum)}")
IO.puts("KL with reduction :none: #{inspect(Nx.to_flat_list(kl_none))}")
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

# ============================================================================
# Orthogonality Axis Options
# ============================================================================

IO.puts("--- Orthogonality Axis Options ---")
IO.puts("")

# 3D tensor: [batch=2, seq=4, vocab=8]
{tensor_3d, _key} = Nx.Random.uniform(Nx.Random.key(42), shape: {2, 4, 8})

# Default (rows): flattens to 2D and orthogonalizes
rows_penalty = Constraints.orthogonality(tensor_3d, axis: :rows)
IO.puts("Orthogonality (axis: :rows): #{Nx.to_number(rows_penalty)}")

# Sequence: orthogonalize across 4 sequence positions
seq_penalty = Constraints.orthogonality(tensor_3d, axis: :sequence)
IO.puts("Orthogonality (axis: :sequence): #{Nx.to_number(seq_penalty)}")

# Vocabulary: orthogonalize across 8 vocabulary dimensions
vocab_penalty = Constraints.orthogonality(tensor_3d, axis: :vocabulary)
IO.puts("Orthogonality (axis: :vocabulary): #{Nx.to_number(vocab_penalty)}")
IO.puts("")

IO.puts("Use case: Token representation diversity")

IO.puts("""
If training embeddings where tensor is [batch, seq, embedding_dim]:
- axis: :sequence encourages different sequence positions to be uncorrelated
- axis: :vocabulary encourages different embedding dimensions to be uncorrelated

For language models:
- axis: :vocabulary on logits encourages different vocabulary tokens
  to have distinct representations
""")
