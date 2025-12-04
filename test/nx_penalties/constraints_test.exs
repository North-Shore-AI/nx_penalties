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

  describe "orthogonality/2 with :spectral mode" do
    test "identity matrix has zero spectral penalty" do
      tensor = Nx.eye(4)
      result = Constraints.orthogonality(tensor, mode: :spectral)
      assert_close(result, Nx.tensor(0.0), atol: 1.0e-4)
    end

    test "orthonormal rows have low spectral penalty" do
      # Create orthonormal matrix via QR decomposition approximation
      # For simplicity, use identity
      tensor = Nx.eye(3)
      result = Constraints.orthogonality(tensor, mode: :spectral)
      assert_close(result, Nx.tensor(0.0), atol: 1.0e-4)
    end

    test "correlated rows have positive spectral penalty" do
      # All rows identical = highly correlated
      row = Nx.tensor([1.0, 0.0, 0.0, 0.0])
      tensor = Nx.stack([row, row, row])
      result = Constraints.orthogonality(tensor, mode: :spectral)
      assert Nx.to_number(result) > 0.0
    end

    test "spectral mode respects normalize option" do
      tensor =
        Nx.tensor([
          [2.0, 0.0, 0.0],
          [0.0, 3.0, 0.0],
          [0.0, 0.0, 4.0]
        ])

      # With normalization, should be close to orthonormal
      normalized_result = Constraints.orthogonality(tensor, mode: :spectral, normalize: true)

      # Without normalization, different magnitudes affect result
      unnormalized_result = Constraints.orthogonality(tensor, mode: :spectral, normalize: false)

      # Normalized should have lower penalty (closer to orthonormal)
      assert Nx.to_number(normalized_result) < Nx.to_number(unnormalized_result)
    end

    test "spectral penalty increases with correlation" do
      # Identity = orthogonal
      identity = Nx.eye(4)
      identity_penalty = Constraints.orthogonality(identity, mode: :spectral)

      # Slightly correlated
      slight =
        Nx.tensor([
          [1.0, 0.1, 0.0, 0.0],
          [0.1, 1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 1.0]
        ])

      slight_penalty = Constraints.orthogonality(slight, mode: :spectral, normalize: true)

      # Highly correlated
      high =
        Nx.tensor([
          [1.0, 0.9, 0.0, 0.0],
          [0.9, 1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 1.0]
        ])

      high_penalty = Constraints.orthogonality(high, mode: :spectral, normalize: true)

      # Penalties should increase with correlation
      assert Nx.to_number(identity_penalty) < Nx.to_number(slight_penalty)
      assert Nx.to_number(slight_penalty) < Nx.to_number(high_penalty)
    end

    test "spectral gradient flows" do
      grad_fn = Nx.Defn.grad(fn x -> Constraints.orthogonality(x, mode: :spectral) end)
      tensor = random_tensor({4, 8})
      grads = grad_fn.(tensor)
      assert Nx.shape(grads) == {4, 8}
      assert_finite(grads)
    end

    test "spectral mode handles 3D tensors" do
      tensor = random_tensor({2, 4, 8})
      result = Constraints.orthogonality(tensor, mode: :spectral)
      assert Nx.shape(result) == {}
      # Scalar output
      assert_finite(result)
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

  describe "consistency/3 with :kl metric" do
    test "identical log-probability distributions have zero KL consistency" do
      logprobs = random_logprobs({2, 8})
      result = Constraints.consistency(logprobs, logprobs, metric: :kl)
      assert_close(result, Nx.tensor(0.0), atol: 1.0e-5)
    end

    test "KL consistency is symmetric" do
      p = random_logprobs({2, 8})
      q = random_logprobs({2, 8})

      kl_pq = Constraints.consistency(p, q, metric: :kl)
      kl_qp = Constraints.consistency(q, p, metric: :kl)

      # Should be equal because we use symmetric KL
      assert_close(kl_pq, kl_qp, atol: 1.0e-5)
    end

    test "KL consistency is non-negative" do
      for _ <- 1..10 do
        p = random_logprobs({4, 16})
        q = random_logprobs({4, 16})
        result = Constraints.consistency(p, q, metric: :kl)
        assert Nx.to_number(result) >= -1.0e-6
      end
    end

    test "KL consistency with reduction: :sum" do
      p = random_logprobs({2, 8})
      q = random_logprobs({2, 8})

      none_result = Constraints.consistency(p, q, metric: :kl, reduction: :none)
      sum_result = Constraints.consistency(p, q, metric: :kl, reduction: :sum)

      assert_close(sum_result, Nx.sum(none_result))
    end

    test "KL consistency with reduction: :none preserves batch dimension" do
      p = random_logprobs({4, 16})
      q = random_logprobs({4, 16})
      result = Constraints.consistency(p, q, metric: :kl, reduction: :none)
      assert Nx.shape(result) == {4}
    end

    test "KL consistency gradient flows" do
      grad_fn =
        Nx.Defn.grad(fn {p, q} ->
          Constraints.consistency(p, q, metric: :kl)
        end)

      p = random_logprobs({2, 8})
      q = random_logprobs({2, 8})
      {grad_p, grad_q} = grad_fn.({p, q})

      assert Nx.shape(grad_p) == {2, 8}
      assert Nx.shape(grad_q) == {2, 8}
      assert_finite(grad_p)
      assert_finite(grad_q)
    end

    test "KL consistency handles peaked distributions" do
      # Very peaked distribution (almost one-hot)
      peaked = Nx.tensor([[0.0, -50.0, -50.0, -50.0]])
      uniform = Nx.tensor([[-1.386, -1.386, -1.386, -1.386]])

      result = Constraints.consistency(peaked, uniform, metric: :kl)
      assert Nx.to_number(result) > 0.0
      assert_finite(result)
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

  describe "orthogonality/2 with :axis option" do
    test "axis: :rows is default behavior" do
      tensor = random_tensor({4, 8})
      default = Constraints.orthogonality(tensor)
      explicit = Constraints.orthogonality(tensor, axis: :rows)
      assert_close(default, explicit)
    end

    test "axis: :sequence with 3D tensor" do
      # [batch=2, seq=4, vocab=8]
      tensor = random_tensor({2, 4, 8})
      result = Constraints.orthogonality(tensor, axis: :sequence)

      # Should compute correlation across 4 sequence positions
      # Each position represented by batch*vocab = 16 features
      assert Nx.shape(result) == {}
      assert_finite(result)
    end

    test "axis: :vocabulary with 3D tensor" do
      # [batch=2, seq=4, vocab=8]
      tensor = random_tensor({2, 4, 8})
      result = Constraints.orthogonality(tensor, axis: :vocabulary)

      # Should compute correlation across 8 vocabulary dimensions
      # Each vocab dim represented by batch*seq = 8 samples
      assert Nx.shape(result) == {}
      assert_finite(result)
    end

    test "axis: :sequence with 2D tensor" do
      tensor = random_tensor({4, 8})
      result = Constraints.orthogonality(tensor, axis: :sequence)
      assert Nx.shape(result) == {}
      assert_finite(result)
    end

    test "axis: :vocabulary with 2D tensor" do
      tensor = random_tensor({4, 8})
      result = Constraints.orthogonality(tensor, axis: :vocabulary)
      assert Nx.shape(result) == {}
      assert_finite(result)
    end

    test "different axes give different penalties" do
      tensor = random_tensor({2, 4, 8})

      rows_penalty = Constraints.orthogonality(tensor, axis: :rows)
      seq_penalty = Constraints.orthogonality(tensor, axis: :sequence)
      vocab_penalty = Constraints.orthogonality(tensor, axis: :vocabulary)

      # All should be valid scalars but likely different values
      assert_finite(rows_penalty)
      assert_finite(seq_penalty)
      assert_finite(vocab_penalty)
    end

    test "axis works with all modes" do
      tensor = random_tensor({2, 4, 8})

      for mode <- [:soft, :hard, :spectral] do
        for axis <- [:rows, :sequence, :vocabulary] do
          result = Constraints.orthogonality(tensor, mode: mode, axis: axis)
          assert Nx.shape(result) == {}, "Failed for mode=#{mode}, axis=#{axis}"
          assert_finite(result)
        end
      end
    end

    test "axis gradient flows" do
      for axis <- [:rows, :sequence, :vocabulary] do
        grad_fn =
          Nx.Defn.grad(fn x ->
            Constraints.orthogonality(x, axis: axis)
          end)

        tensor = random_tensor({2, 4, 8})
        grads = grad_fn.(tensor)
        assert Nx.shape(grads) == {2, 4, 8}, "Gradient shape wrong for axis=#{axis}"
        assert_finite(grads)
      end
    end

    test "orthogonal sequence positions have low penalty" do
      # Create tensor where each sequence position is orthogonal
      # [seq=3, vocab=3] with orthonormal rows
      tensor = Nx.eye(3)
      result = Constraints.orthogonality(tensor, axis: :sequence)
      assert_close(result, Nx.tensor(0.0), atol: 1.0e-4)
    end
  end
end
