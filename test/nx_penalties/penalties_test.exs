defmodule NxPenalties.PenaltiesTest do
  use ExUnit.Case, async: true

  import NxPenalties.TestHelpers
  import NxPenalties.NumericalHelpers

  alias NxPenalties.Penalties

  describe "l1/2" do
    test "returns scalar for any input shape" do
      for shape <- [{4}, {2, 3}, {2, 3, 4}] do
        tensor = random_tensor(shape)
        result = Penalties.l1(tensor)
        assert_scalar(result)
      end
    end

    test "computes correct L1 norm with default lambda" do
      tensor = Nx.tensor([1.0, -2.0, 3.0])
      result = Penalties.l1(tensor)
      # Expected: 1.0 * (1 + 2 + 3) = 6.0 (unscaled by default)
      assert_close(result, Nx.tensor(6.0))
    end

    test "respects custom lambda" do
      tensor = Nx.tensor([1.0, -1.0])
      result = Penalties.l1(tensor, lambda: 0.5)
      # Expected: 0.5 * 2 = 1.0
      assert_close(result, Nx.tensor(1.0))
    end

    test "handles zero values" do
      tensor = Nx.tensor([0.0, 0.0, 1.0])
      result = Penalties.l1(tensor, lambda: 1.0)
      assert_close(result, Nx.tensor(1.0))
    end

    test "mean reduction" do
      tensor = Nx.tensor([1.0, 2.0, 3.0])
      result = Penalties.l1(tensor, lambda: 1.0, reduction: :mean)
      assert_close(result, Nx.tensor(2.0))
    end

    test "works with multidimensional tensors" do
      tensor = Nx.tensor([[1.0, -2.0], [-3.0, 4.0]])
      result = Penalties.l1(tensor, lambda: 0.1)
      # Expected: 0.1 * 10 = 1.0
      assert_close(result, Nx.tensor(1.0))
    end

    test "gradient is sign function" do
      grad_fn = Nx.Defn.grad(fn x -> Penalties.l1(x, lambda: 1.0) end)
      tensor = Nx.tensor([2.0, -3.0, 0.5])
      grads = grad_fn.(tensor)
      # For non-zero values: sign(x)
      assert_close(grads, Nx.tensor([1.0, -1.0, 1.0]))
    end

    test "JIT compiles successfully" do
      tensor = random_tensor({10})
      assert_jit_compiles(&Penalties.l1/2, [tensor, [lambda: 0.1]])
    end

    test "numerically stable across value ranges" do
      check_stability(
        fn x -> Penalties.l1(x, lambda: 0.01) end,
        [
          {-1.0e-10, 1.0e-10, "very small"},
          {-1.0, 1.0, "normal"},
          {-1.0e6, 1.0e6, "large"}
        ]
      )
    end
  end

  describe "l2/2" do
    test "returns scalar for any input shape" do
      for shape <- [{4}, {2, 3}, {2, 3, 4}] do
        tensor = random_tensor(shape)
        result = Penalties.l2(tensor)
        assert_scalar(result)
      end
    end

    test "computes correct L2 norm" do
      tensor = Nx.tensor([1.0, 2.0, 3.0])
      result = Penalties.l2(tensor, lambda: 0.1)
      # Expected: 0.1 * (1 + 4 + 9) = 1.4
      assert_close(result, Nx.tensor(1.4))
    end

    test "handles negative values (squared)" do
      tensor = Nx.tensor([-2.0, -3.0])
      result = Penalties.l2(tensor, lambda: 0.1)
      # Expected: 0.1 * (4 + 9) = 1.3
      assert_close(result, Nx.tensor(1.3))
    end

    test "clipping prevents overflow" do
      tensor = Nx.tensor([1000.0, 2000.0])
      # Without clip, this would be large
      result = Penalties.l2(tensor, lambda: 0.01, clip: 100.0)
      # Clipped to [100, 100], squared = [10000, 10000], sum * 0.01 = 200
      assert_close(result, Nx.tensor(200.0))
    end

    test "gradient is 2*lambda*x" do
      grad_fn = Nx.Defn.grad(fn x -> Penalties.l2(x, lambda: 0.5) end)
      tensor = Nx.tensor([1.0, 2.0, 3.0])
      grads = grad_fn.(tensor)
      # Expected: 2 * 0.5 * [1, 2, 3] = [1, 2, 3]
      assert_close(grads, Nx.tensor([1.0, 2.0, 3.0]))
    end

    test "mean reduction" do
      tensor = Nx.tensor([1.0, 2.0])
      result = Penalties.l2(tensor, lambda: 1.0, reduction: :mean)
      # Expected: mean([1, 4]) = 2.5
      assert_close(result, Nx.tensor(2.5))
    end

    test "JIT compiles successfully" do
      tensor = random_tensor({10})
      assert_jit_compiles(&Penalties.l2/2, [tensor, [lambda: 0.1]])
    end

    test "returns zero for zero tensor" do
      tensor = Nx.tensor([0.0, 0.0, 0.0])
      result = Penalties.l2(tensor, lambda: 1.0)
      assert_close(result, Nx.tensor(0.0))
    end
  end

  describe "elastic_net/2" do
    test "returns scalar for any input shape" do
      for shape <- [{4}, {2, 3}] do
        tensor = random_tensor(shape)
        result = Penalties.elastic_net(tensor)
        assert_scalar(result)
      end
    end

    test "with l1_ratio=1.0 equals l1" do
      tensor = Nx.tensor([1.0, -2.0, 3.0])
      elastic = Penalties.elastic_net(tensor, lambda: 0.1, l1_ratio: 1.0)
      l1 = Penalties.l1(tensor, lambda: 0.1)
      assert_close(elastic, l1)
    end

    test "with l1_ratio=0.0 equals l2" do
      tensor = Nx.tensor([1.0, -2.0, 3.0])
      elastic = Penalties.elastic_net(tensor, lambda: 0.1, l1_ratio: 0.0)
      l2 = Penalties.l2(tensor, lambda: 0.1)
      assert_close(elastic, l2)
    end

    test "balanced ratio combines both" do
      tensor = Nx.tensor([1.0, 2.0])
      result = Penalties.elastic_net(tensor, lambda: 1.0, l1_ratio: 0.5)
      # L1: 3, L2: 5, combined: 0.5*3 + 0.5*5 = 4.0
      assert_close(result, Nx.tensor(4.0))
    end

    test "gradient combines L1 and L2 gradients" do
      grad_fn =
        Nx.Defn.grad(fn x ->
          Penalties.elastic_net(x, lambda: 1.0, l1_ratio: 0.5)
        end)

      tensor = Nx.tensor([2.0, -3.0])
      grads = grad_fn.(tensor)
      # L1 grad: [1, -1] * 0.5 = [0.5, -0.5]
      # L2 grad: [4, -6] * 0.5 = [2, -3]
      # Combined: [2.5, -3.5]
      assert_close(grads, Nx.tensor([2.5, -3.5]))
    end

    test "JIT compiles successfully" do
      tensor = random_tensor({10})
      assert_jit_compiles(&Penalties.elastic_net/2, [tensor, [lambda: 0.1, l1_ratio: 0.5]])
    end

    test "mean reduction" do
      tensor = Nx.tensor([1.0, 2.0, 3.0])
      result = Penalties.elastic_net(tensor, lambda: 1.0, l1_ratio: 0.5, reduction: :mean)
      # L1 mean: 2, L2 mean: (1+4+9)/3 = 14/3
      # Combined: 0.5 * 2 + 0.5 * 14/3 = 1 + 7/3 = 10/3
      assert_close(result, Nx.tensor(10.0 / 3.0))
    end
  end

  describe "numerical stability" do
    test "l1 handles very small values" do
      tensor = Nx.tensor([1.0e-30, 1.0e-30])
      result = Penalties.l1(tensor)
      assert_finite(result)
    end

    test "l2 handles very small values" do
      tensor = Nx.tensor([1.0e-30, 1.0e-30])
      result = Penalties.l2(tensor)
      assert_finite(result)
    end

    test "l1 gradient is finite at zero" do
      grad_fn = Nx.Defn.grad(fn x -> Penalties.l1(x) end)
      grads = grad_fn.(Nx.tensor([0.0, 0.0, 0.0]))
      assert_finite(grads)
    end

    test "l2 gradient is finite at zero" do
      grad_fn = Nx.Defn.grad(fn x -> Penalties.l2(x) end)
      grads = grad_fn.(Nx.tensor([0.0, 0.0, 0.0]))
      assert_finite(grads)
    end
  end

  describe "validate/1" do
    test "returns {:ok, tensor} for finite tensor" do
      tensor = Nx.tensor([1.0, 2.0, 3.0])
      assert {:ok, ^tensor} = NxPenalties.validate(tensor)
    end

    test "returns {:error, :nan} for tensor with NaN" do
      tensor = Nx.Constants.nan({:f, 32})
      assert {:error, :nan} = NxPenalties.validate(tensor)
    end

    test "returns {:error, :inf} for tensor with infinity" do
      tensor = Nx.Constants.infinity({:f, 32})
      assert {:error, :inf} = NxPenalties.validate(tensor)
    end

    test "returns {:error, :inf} for tensor with negative infinity" do
      tensor = Nx.Constants.neg_infinity({:f, 32})
      assert {:error, :inf} = NxPenalties.validate(tensor)
    end

    test "returns {:ok, tensor} for multidimensional tensors" do
      tensor = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      assert {:ok, ^tensor} = NxPenalties.validate(tensor)
    end

    test "detects NaN in multidimensional tensor" do
      # Use Nx.select to embed a NaN in a tensor
      nan = Nx.Constants.nan({:f, 32})
      base = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      mask = Nx.tensor([[0, 1], [0, 0]], type: :u8)
      tensor = Nx.select(mask, nan, base)
      assert {:error, :nan} = NxPenalties.validate(tensor)
    end

    test "detects Inf in multidimensional tensor" do
      inf = Nx.Constants.infinity({:f, 32})
      base = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      mask = Nx.tensor([[1, 0], [0, 0]], type: :u8)
      tensor = Nx.select(mask, inf, base)
      assert {:error, :inf} = NxPenalties.validate(tensor)
    end
  end
end
