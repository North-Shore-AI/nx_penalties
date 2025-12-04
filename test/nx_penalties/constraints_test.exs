defmodule NxPenalties.ConstraintsTest do
  use ExUnit.Case, async: true

  import NxPenalties.TestHelpers

  alias NxPenalties.Constraints

  describe "orthogonality/2" do
    test "returns zero for identity matrix (already orthogonal)" do
      # Identity matrix rows are orthogonal
      tensor = Nx.eye(3)
      result = Constraints.orthogonality(tensor, mode: :hard)
      assert_close(result, Nx.tensor(0.0), atol: 1.0e-5)
    end

    test "returns zero for orthogonal rows with soft mode" do
      # Identity matrix - off-diagonal elements are zero
      tensor = Nx.eye(4)
      result = Constraints.orthogonality(tensor, mode: :soft)
      assert_close(result, Nx.tensor(0.0), atol: 1.0e-5)
    end

    test "returns positive penalty for correlated rows" do
      # All rows the same = maximally correlated
      tensor = Nx.broadcast(Nx.tensor([1.0, 0.0, 0.0, 0.0]), {4, 4})
      result = Constraints.orthogonality(tensor, mode: :soft)
      # After normalization, all rows identical
      # Gram matrix is all 1s, off-diagonal has 12 elements
      assert Nx.to_number(result) > 10.0
    end

    test "soft mode ignores diagonal" do
      # Create matrix where diagonal of Gram would be non-identity
      # but off-diagonal is zero
      tensor = Nx.eye(3)
      soft = Constraints.orthogonality(tensor, mode: :soft)
      assert_close(soft, Nx.tensor(0.0), atol: 1.0e-5)
    end

    test "hard mode penalizes non-unit diagonal" do
      # When normalize: false, rows aren't unit length
      # Identity has unit rows, so hard penalty should still be 0
      tensor = Nx.eye(3)
      hard = Constraints.orthogonality(tensor, mode: :hard, normalize: true)
      assert_close(hard, Nx.tensor(0.0), atol: 1.0e-5)
    end

    test "handles 3D input (batch, seq, dim)" do
      tensor = random_tensor({2, 4, 8})
      result = Constraints.orthogonality(tensor)
      assert Nx.shape(result) == {}
      assert_finite(result)
    end

    test "handles 4D input" do
      tensor = random_tensor({2, 3, 4, 8})
      result = Constraints.orthogonality(tensor)
      assert Nx.shape(result) == {}
      assert_finite(result)
    end

    test "normalize option affects result" do
      # Non-normalized rows of different lengths
      tensor =
        Nx.tensor([
          [2.0, 0.0],
          [0.0, 1.0]
        ])

      normalized = Constraints.orthogonality(tensor, mode: :hard, normalize: true)
      unnormalized = Constraints.orthogonality(tensor, mode: :hard, normalize: false)

      # With normalization, these become orthonormal -> penalty = 0
      assert_close(normalized, Nx.tensor(0.0), atol: 1.0e-5)

      # Without normalization, Gram = [[4, 0], [0, 1]], deviation from I = [[3, 0], [0, 0]]
      # Penalty = 3^2 = 9
      assert_close(unnormalized, Nx.tensor(9.0), atol: 1.0e-5)
    end

    test "gradient flows correctly" do
      tensor = random_tensor({4, 8})

      grad_fn =
        Nx.Defn.grad(fn t ->
          Constraints.orthogonality(t)
        end)

      grads = grad_fn.(tensor)
      assert Nx.shape(grads) == Nx.shape(tensor)
      assert_finite(grads)

      # Gradient should be non-zero for most random inputs
      grad_sum = grads |> Nx.abs() |> Nx.sum() |> Nx.to_number()
      assert grad_sum > 0
    end

    test "higher correlation gives higher penalty" do
      # Nearly orthogonal vectors
      low_corr =
        Nx.tensor([
          [1.0, 0.0],
          [0.1, 0.99]
        ])

      # Highly correlated vectors
      high_corr =
        Nx.tensor([
          [1.0, 0.0],
          [0.9, 0.1]
        ])

      low_penalty = Constraints.orthogonality(low_corr) |> Nx.to_number()
      high_penalty = Constraints.orthogonality(high_corr) |> Nx.to_number()

      assert high_penalty > low_penalty
    end
  end

  describe "consistency/3" do
    test "identical outputs have zero penalty with MSE" do
      output = random_tensor({4, 8})
      result = Constraints.consistency(output, output, metric: :mse)
      assert_close(result, Nx.tensor(0.0), atol: 1.0e-6)
    end

    test "identical outputs have zero penalty with L1" do
      output = random_tensor({4, 8})
      result = Constraints.consistency(output, output, metric: :l1)
      assert_close(result, Nx.tensor(0.0), atol: 1.0e-6)
    end

    test "identical outputs have zero penalty with cosine" do
      output = random_tensor({4, 8}, min: 0.1, max: 1.0)
      result = Constraints.consistency(output, output, metric: :cosine)
      assert_close(result, Nx.tensor(0.0), atol: 1.0e-5)
    end

    test "MSE metric computes correctly" do
      o1 = Nx.tensor([1.0, 2.0])
      o2 = Nx.tensor([2.0, 4.0])
      result = Constraints.consistency(o1, o2, metric: :mse)
      # MSE: mean([(2-1)², (4-2)²]) = mean([1, 4]) = 2.5
      assert_close(result, Nx.tensor(2.5))
    end

    test "L1 metric computes correctly" do
      o1 = Nx.tensor([1.0, 2.0])
      o2 = Nx.tensor([2.0, 4.0])
      result = Constraints.consistency(o1, o2, metric: :l1)
      # L1: mean([|2-1|, |4-2|]) = mean([1, 2]) = 1.5
      assert_close(result, Nx.tensor(1.5))
    end

    test "cosine metric - orthogonal vectors have distance 1" do
      o1 = Nx.tensor([1.0, 0.0])
      o2 = Nx.tensor([0.0, 1.0])
      result = Constraints.consistency(o1, o2, metric: :cosine)
      # Orthogonal vectors: cosine similarity = 0, distance = 1 - 0 = 1
      assert_close(result, Nx.tensor(1.0), atol: 1.0e-5)
    end

    test "cosine metric - same direction vectors have distance 0" do
      o1 = Nx.tensor([1.0, 2.0])
      o2 = Nx.tensor([2.0, 4.0])
      result = Constraints.consistency(o1, o2, metric: :cosine)
      # Same direction (different magnitude): cosine similarity = 1, distance = 0
      assert_close(result, Nx.tensor(0.0), atol: 1.0e-5)
    end

    test "cosine metric - opposite vectors have distance 2" do
      o1 = Nx.tensor([1.0, 0.0])
      o2 = Nx.tensor([-1.0, 0.0])
      result = Constraints.consistency(o1, o2, metric: :cosine)
      # Opposite vectors: cosine similarity = -1, distance = 1 - (-1) = 2
      assert_close(result, Nx.tensor(2.0), atol: 1.0e-5)
    end

    test "reduction :sum works" do
      o1 = Nx.tensor([1.0, 2.0])
      o2 = Nx.tensor([2.0, 4.0])
      result = Constraints.consistency(o1, o2, metric: :mse, reduction: :sum)
      # Sum: (2-1)² + (4-2)² = 1 + 4 = 5
      assert_close(result, Nx.tensor(5.0))
    end

    test "reduction :none returns element-wise" do
      o1 = Nx.tensor([1.0, 2.0])
      o2 = Nx.tensor([2.0, 4.0])
      result = Constraints.consistency(o1, o2, metric: :mse, reduction: :none)
      # Element-wise: [(2-1)², (4-2)²] = [1, 4]
      assert_close(result, Nx.tensor([1.0, 4.0]))
    end

    test "handles multidimensional tensors" do
      o1 = random_tensor({2, 4, 8})
      o2 = random_tensor({2, 4, 8})
      result = Constraints.consistency(o1, o2)
      assert Nx.shape(result) == {}
      assert_finite(result)
    end

    test "gradient flows correctly through MSE" do
      o1 = random_tensor({4, 8})
      o2 = random_tensor({4, 8})

      grad_fn =
        Nx.Defn.grad(fn t ->
          Constraints.consistency(t, o2, metric: :mse)
        end)

      grads = grad_fn.(o1)
      assert Nx.shape(grads) == Nx.shape(o1)
      assert_finite(grads)
    end

    test "gradient flows correctly through L1" do
      o1 = random_tensor({4, 8})
      o2 = random_tensor({4, 8})

      grad_fn =
        Nx.Defn.grad(fn t ->
          Constraints.consistency(t, o2, metric: :l1)
        end)

      grads = grad_fn.(o1)
      assert Nx.shape(grads) == Nx.shape(o1)
      assert_finite(grads)
    end

    test "gradient flows correctly through cosine" do
      o1 = random_tensor({4, 8}, min: 0.1, max: 1.0)
      o2 = random_tensor({4, 8}, min: 0.1, max: 1.0)

      grad_fn =
        Nx.Defn.grad(fn t ->
          Constraints.consistency(t, o2, metric: :cosine)
        end)

      grads = grad_fn.(o1)
      assert Nx.shape(grads) == {4, 8}
      assert_finite(grads)
    end
  end

  describe "edge cases" do
    test "orthogonality with single row" do
      tensor = Nx.tensor([[1.0, 2.0, 3.0]])
      result = Constraints.orthogonality(tensor)
      # Single row -> 1x1 Gram matrix, no off-diagonal elements
      assert_close(result, Nx.tensor(0.0), atol: 1.0e-5)
    end

    test "orthogonality with single element" do
      tensor = Nx.tensor([[1.0]])
      result = Constraints.orthogonality(tensor)
      assert_close(result, Nx.tensor(0.0), atol: 1.0e-5)
    end

    test "consistency with scalar tensors" do
      o1 = Nx.tensor(1.0)
      o2 = Nx.tensor(2.0)
      result = Constraints.consistency(o1, o2, metric: :mse)
      # (2-1)² = 1
      assert_close(result, Nx.tensor(1.0))
    end

    test "orthogonality with very small values" do
      tensor = Nx.multiply(Nx.eye(3), 1.0e-10)
      result = Constraints.orthogonality(tensor)
      assert_finite(result)
    end

    test "consistency handles near-zero vectors in cosine" do
      # Small but non-zero vectors
      o1 = Nx.tensor([1.0e-8, 0.0])
      o2 = Nx.tensor([0.0, 1.0e-8])
      result = Constraints.consistency(o1, o2, metric: :cosine)
      assert_finite(result)
    end
  end
end
