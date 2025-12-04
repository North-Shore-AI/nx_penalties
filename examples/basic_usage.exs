# Basic Usage Examples
#
# Run with: mix run examples/basic_usage.exs

IO.puts("=== NxPenalties Basic Usage ===\n")

# Sample tensor
tensor = Nx.tensor([1.0, -2.0, 3.0, -0.5])
IO.puts("Input tensor: #{inspect(Nx.to_flat_list(tensor))}")
IO.puts("")

# L1 Penalty (Lasso)
l1 = NxPenalties.l1(tensor)
IO.puts("L1 penalty (lambda=1.0): #{Nx.to_number(l1)}")
# Expected: |1| + |-2| + |3| + |-0.5| = 6.5

l1_scaled = NxPenalties.l1(tensor, lambda: 0.1)
IO.puts("L1 penalty (lambda=0.1): #{Nx.to_number(l1_scaled)}")
# Expected: 0.65

IO.puts("")

# L2 Penalty (Ridge)
l2 = NxPenalties.l2(tensor)
IO.puts("L2 penalty (lambda=1.0): #{Nx.to_number(l2)}")
# Expected: 1 + 4 + 9 + 0.25 = 14.25

l2_scaled = NxPenalties.l2(tensor, lambda: 0.01)
IO.puts("L2 penalty (lambda=0.01): #{Nx.to_number(l2_scaled)}")
# Expected: 0.1425

IO.puts("")

# Elastic Net (Combined L1 + L2)
elastic = NxPenalties.elastic_net(tensor, l1_ratio: 0.5)
IO.puts("Elastic Net (l1_ratio=0.5): #{Nx.to_number(elastic)}")
# Expected: 0.5 * 6.5 + 0.5 * 14.25 = 10.375

elastic_l1 = NxPenalties.elastic_net(tensor, l1_ratio: 1.0)
IO.puts("Elastic Net (l1_ratio=1.0, pure L1): #{Nx.to_number(elastic_l1)}")

elastic_l2 = NxPenalties.elastic_net(tensor, l1_ratio: 0.0)
IO.puts("Elastic Net (l1_ratio=0.0, pure L2): #{Nx.to_number(elastic_l2)}")

IO.puts("")

# Reduction modes
IO.puts("--- Reduction Modes ---")
l1_sum = NxPenalties.l1(tensor, reduction: :sum)
l1_mean = NxPenalties.l1(tensor, reduction: :mean)
IO.puts("L1 sum: #{Nx.to_number(l1_sum)}")
IO.puts("L1 mean: #{Nx.to_number(l1_mean)}")

IO.puts("\n=== Done ===")
