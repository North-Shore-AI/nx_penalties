# Entropy Normalization Examples
#
# Run with: mix run examples/entropy_normalization.exs

IO.puts("=== Entropy Normalization ===\n")

# Example 1: Uniform distribution over different vocabulary sizes
IO.puts("--- Comparing Uniform Distributions ---")

for vocab_size <- [10, 100, 1000] do
  uniform = Nx.broadcast(Nx.tensor(-:math.log(vocab_size)), {1, vocab_size})

  raw_entropy = NxPenalties.entropy(uniform, normalize: false)
  normalized_entropy = NxPenalties.entropy(uniform, normalize: true)

  IO.puts("\nVocabulary size: #{vocab_size}")
  IO.puts("  Raw entropy: #{Nx.to_number(raw_entropy) |> Float.round(4)}")
  IO.puts("  Normalized entropy: #{Nx.to_number(normalized_entropy) |> Float.round(4)}")
  IO.puts("  Expected raw: #{:math.log(vocab_size) |> Float.round(4)} (log(#{vocab_size}))")
  IO.puts("  Expected normalized: 1.0 (maximum uncertainty)")
end

# Example 2: Peaked distributions
IO.puts("\n\n--- Peaked Distributions ---")

# Almost all mass on first token
peaked = Nx.tensor([[-0.01, -5.0, -5.0, -5.0]])
peaked_raw = NxPenalties.entropy(peaked, normalize: false)
peaked_normalized = NxPenalties.entropy(peaked, normalize: true)

IO.puts("\nPeaked distribution (almost one-hot):")
IO.puts("  Raw entropy: #{Nx.to_number(peaked_raw) |> Float.round(4)}")
IO.puts("  Normalized entropy: #{Nx.to_number(peaked_normalized) |> Float.round(4)}")
IO.puts("  Expected: Close to 0.0 (low uncertainty)")

# Example 3: Comparing distributions across different vocab sizes
IO.puts("\n\n--- Cross-Vocabulary Comparison ---")

# Small vocab (4 tokens) - peaked
small_vocab_peaked = Nx.tensor([[0.0, -5.0, -5.0, -5.0]])
# Large vocab (100 tokens) - peaked
large_vocab_peaked =
  Nx.concatenate(
    [
      Nx.tensor([[0.0]]),
      Nx.broadcast(Nx.tensor(-5.0), {1, 99})
    ],
    axis: 1
  )

small_norm = NxPenalties.entropy(small_vocab_peaked, normalize: true)
large_norm = NxPenalties.entropy(large_vocab_peaked, normalize: true)

IO.puts("\nPeaked distribution with vocab_size=4:")
IO.puts("  Normalized entropy: #{Nx.to_number(small_norm) |> Float.round(4)}")
IO.puts("\nPeaked distribution with vocab_size=100:")
IO.puts("  Normalized entropy: #{Nx.to_number(large_norm) |> Float.round(4)}")
IO.puts("\nNormalization allows fair comparison across different vocabulary sizes!")

# Example 4: Using with penalty mode
IO.puts("\n\n--- Penalty Mode ---")

uniform_4 = Nx.broadcast(Nx.tensor(-:math.log(4)), {1, 4})

bonus = NxPenalties.entropy(uniform_4, normalize: true, mode: :bonus)
penalty = NxPenalties.entropy(uniform_4, normalize: true, mode: :penalty)

IO.puts("\nUniform distribution (4 tokens):")
IO.puts("  Bonus mode (encourages high entropy): #{Nx.to_number(bonus) |> Float.round(4)}")
IO.puts("  Penalty mode (discourages high entropy): #{Nx.to_number(penalty) |> Float.round(4)}")

# Example 5: Batch processing with reduction modes
IO.puts("\n\n--- Batch Processing ---")

# Create a batch of 3 distributions with varying entropy
batch =
  Nx.tensor([
    # Peaked (low entropy)
    [0.0, -5.0, -5.0, -5.0],
    # Uniform (high entropy)
    [-1.386, -1.386, -1.386, -1.386],
    # Medium entropy
    [-0.5, -1.5, -2.0, -3.0]
  ])

mean_result = NxPenalties.entropy(batch, normalize: true, reduction: :mean)
sum_result = NxPenalties.entropy(batch, normalize: true, reduction: :sum)
none_result = NxPenalties.entropy(batch, normalize: true, reduction: :none)

IO.puts("\nBatch of 3 distributions:")
IO.puts("  Mean reduction: #{Nx.to_number(mean_result) |> Float.round(4)}")
IO.puts("  Sum reduction: #{Nx.to_number(sum_result) |> Float.round(4)}")

IO.puts(
  "  None reduction (per-sample): #{inspect(Nx.to_flat_list(none_result) |> Enum.map(&Float.round(&1, 4)))}"
)

IO.puts("\n=== Done ===")
