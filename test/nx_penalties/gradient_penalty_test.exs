defmodule NxPenalties.GradientPenaltyTest do
  use ExUnit.Case, async: true

  import NxPenalties.TestHelpers

  alias NxPenalties.GradientPenalty

  describe "gradient_penalty/3" do
    test "returns scalar penalty" do
      loss_fn = fn x -> Nx.sum(x) end
      tensor = Nx.tensor([1.0, 2.0, 3.0])
      result = GradientPenalty.gradient_penalty(loss_fn, tensor)
      assert Nx.shape(result) == {}
    end

    test "penalty is zero when gradient norm equals target" do
      # Gradient of sum(x) is all ones, norm = sqrt(n)
      loss_fn = fn x -> Nx.sum(x) end
      tensor = Nx.tensor([1.0, 2.0, 3.0])
      # grad = [1,1,1], norm = sqrt(3)

      result = GradientPenalty.gradient_penalty(loss_fn, tensor, target_norm: :math.sqrt(3))
      assert_close(result, Nx.tensor(0.0), atol: 1.0e-4)
    end

    test "penalty increases with deviation from target" do
      loss_fn = fn x -> Nx.sum(x) end
      tensor = Nx.tensor([1.0, 1.0, 1.0, 1.0])
      # grad norm = 2

      # Close to actual norm
      penalty_close = GradientPenalty.gradient_penalty(loss_fn, tensor, target_norm: 2.0)

      # Far from actual norm
      penalty_far = GradientPenalty.gradient_penalty(loss_fn, tensor, target_norm: 10.0)

      assert Nx.to_number(penalty_close) < Nx.to_number(penalty_far)
    end

    test "works with L2 loss function" do
      loss_fn = fn x -> Nx.sum(Nx.pow(x, 2)) end
      tensor = Nx.tensor([1.0, 2.0])
      # grad = [2, 4], norm = sqrt(20)

      result = GradientPenalty.gradient_penalty(loss_fn, tensor, target_norm: :math.sqrt(20))
      assert_close(result, Nx.tensor(0.0), atol: 1.0e-4)
    end

    test "handles multidimensional tensors" do
      loss_fn = fn x -> Nx.sum(x) end
      tensor = random_tensor({2, 3, 4})
      result = GradientPenalty.gradient_penalty(loss_fn, tensor)
      assert Nx.shape(result) == {}
      assert_finite(result)
    end
  end

  describe "output_magnitude_penalty/2" do
    test "returns scalar penalty" do
      output = Nx.tensor([1.0, 2.0, 3.0])
      result = GradientPenalty.output_magnitude_penalty(output)
      assert Nx.shape(result) == {}
    end

    test "zero penalty when magnitude equals target" do
      # Create output with known magnitude
      # [1, 0, 0] has magnitude 1
      output = Nx.tensor([1.0, 0.0, 0.0])
      result = GradientPenalty.output_magnitude_penalty(output, target: 1.0, reduction: :sum)
      assert_close(result, Nx.tensor(0.0), atol: 1.0e-4)
    end

    test "penalty increases with magnitude deviation" do
      output = Nx.tensor([3.0, 4.0])
      # magnitude = 5 (for sum), sqrt(12.5) for mean

      # Close to actual magnitude
      penalty_close = GradientPenalty.output_magnitude_penalty(output, target: 3.5)

      # Far from actual magnitude
      penalty_far = GradientPenalty.output_magnitude_penalty(output, target: 10.0)

      assert Nx.to_number(penalty_close) < Nx.to_number(penalty_far)
    end

    test "reduction: :mean uses RMS" do
      output = Nx.tensor([2.0, 2.0, 2.0, 2.0])
      result = GradientPenalty.output_magnitude_penalty(output, target: 2.0, reduction: :mean)
      assert_close(result, Nx.tensor(0.0), atol: 1.0e-4)
    end

    test "reduction: :sum uses L2 norm" do
      output = Nx.tensor([3.0, 4.0])
      # L2 norm = 5
      result = GradientPenalty.output_magnitude_penalty(output, target: 5.0, reduction: :sum)
      assert_close(result, Nx.tensor(0.0), atol: 1.0e-4)
    end

    test "gradient flows" do
      grad_fn =
        Nx.Defn.grad(fn x ->
          GradientPenalty.output_magnitude_penalty(x, target: 1.0)
        end)

      output = random_tensor({4, 8})
      grads = grad_fn.(output)
      assert Nx.shape(grads) == {4, 8}
      assert_finite(grads)
    end
  end

  describe "interpolated_gradient_penalty/4" do
    test "returns scalar penalty" do
      loss_fn = fn x -> Nx.sum(x) end
      tensor = Nx.tensor([1.0, 2.0, 3.0])
      reference = Nx.tensor([0.0, 0.0, 0.0])
      result = GradientPenalty.interpolated_gradient_penalty(loss_fn, tensor, reference)
      assert Nx.shape(result) == {}
    end

    test "penalty is non-negative" do
      loss_fn = fn x -> Nx.sum(Nx.pow(x, 2)) end

      for _ <- 1..5 do
        tensor = random_tensor({4})
        reference = random_tensor({4})
        result = GradientPenalty.interpolated_gradient_penalty(loss_fn, tensor, reference)
        assert Nx.to_number(result) >= 0.0
      end
    end

    test "works with multidimensional tensors" do
      loss_fn = fn x -> Nx.mean(x) end
      tensor = random_tensor({2, 4})
      reference = random_tensor({2, 4})
      result = GradientPenalty.interpolated_gradient_penalty(loss_fn, tensor, reference)
      assert Nx.shape(result) == {}
      assert_finite(result)
    end
  end
end
