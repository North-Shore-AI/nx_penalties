defmodule NxPenalties.PipelineTest do
  use ExUnit.Case, async: true

  import NxPenalties.TestHelpers

  alias NxPenalties.Divergences
  alias NxPenalties.Penalties
  alias NxPenalties.Pipeline

  describe "new/1" do
    test "creates empty pipeline" do
      pipeline = Pipeline.new()
      assert pipeline.entries == []
    end

    test "accepts name option" do
      pipeline = Pipeline.new(name: "my_pipeline")
      assert pipeline.name == "my_pipeline"
    end
  end

  describe "add/4" do
    test "adds penalty to pipeline" do
      pipeline =
        Pipeline.new()
        |> Pipeline.add(:l1, &Penalties.l1/2, weight: 0.01)

      assert length(pipeline.entries) == 1
      [{name, _fn, weight, _opts, enabled}] = pipeline.entries
      assert name == :l1
      assert weight == 0.01
      assert enabled == true
    end

    test "adds multiple penalties" do
      pipeline =
        Pipeline.new()
        |> Pipeline.add(:l1, &Penalties.l1/2, weight: 0.01)
        |> Pipeline.add(:l2, &Penalties.l2/2, weight: 0.001)

      assert length(pipeline.entries) == 2
    end

    test "stores options" do
      pipeline =
        Pipeline.new()
        |> Pipeline.add(:l1, &Penalties.l1/2, weight: 0.01, opts: [reduction: :mean])

      [{_name, _fn, _weight, opts, _enabled}] = pipeline.entries
      assert opts[:reduction] == :mean
    end

    test "weight defaults to 1.0" do
      pipeline =
        Pipeline.new()
        |> Pipeline.add(:l1, &Penalties.l1/2)

      [{_name, _fn, weight, _opts, _enabled}] = pipeline.entries
      assert weight == 1.0
    end
  end

  describe "remove/2" do
    test "removes penalty by name" do
      pipeline =
        Pipeline.new()
        |> Pipeline.add(:l1, &Penalties.l1/2, weight: 0.01)
        |> Pipeline.add(:l2, &Penalties.l2/2, weight: 0.001)
        |> Pipeline.remove(:l1)

      assert length(pipeline.entries) == 1
      [{name, _, _, _, _}] = pipeline.entries
      assert name == :l2
    end

    test "returns unchanged pipeline if name not found" do
      pipeline =
        Pipeline.new()
        |> Pipeline.add(:l1, &Penalties.l1/2, weight: 0.01)
        |> Pipeline.remove(:nonexistent)

      assert length(pipeline.entries) == 1
    end
  end

  describe "update_weight/3" do
    test "updates weight for existing penalty" do
      pipeline =
        Pipeline.new()
        |> Pipeline.add(:l1, &Penalties.l1/2, weight: 0.01)
        |> Pipeline.update_weight(:l1, 0.05)

      [{_name, _fn, weight, _opts, _enabled}] = pipeline.entries
      assert weight == 0.05
    end

    test "accepts tensor weight" do
      weight_tensor = Nx.tensor(0.02)

      pipeline =
        Pipeline.new()
        |> Pipeline.add(:l1, &Penalties.l1/2, weight: 0.01)
        |> Pipeline.update_weight(:l1, weight_tensor)

      [{_name, _fn, weight, _opts, _enabled}] = pipeline.entries
      assert weight == weight_tensor
    end
  end

  describe "set_enabled/3" do
    test "disables penalty" do
      pipeline =
        Pipeline.new()
        |> Pipeline.add(:l1, &Penalties.l1/2, weight: 0.01)
        |> Pipeline.set_enabled(:l1, false)

      [{_name, _fn, _weight, _opts, enabled}] = pipeline.entries
      assert enabled == false
    end

    test "re-enables penalty" do
      pipeline =
        Pipeline.new()
        |> Pipeline.add(:l1, &Penalties.l1/2, weight: 0.01)
        |> Pipeline.set_enabled(:l1, false)
        |> Pipeline.set_enabled(:l1, true)

      [{_name, _fn, _weight, _opts, enabled}] = pipeline.entries
      assert enabled == true
    end
  end

  describe "compute/3" do
    test "returns total and metrics" do
      pipeline =
        Pipeline.new()
        |> Pipeline.add(:l1, &Penalties.l1/2, weight: 0.01)

      tensor = Nx.tensor([1.0, -2.0, 3.0])
      {total, metrics} = Pipeline.compute(pipeline, tensor)

      assert_scalar(total)
      assert is_map(metrics)
    end

    test "computes weighted sum of penalties" do
      pipeline =
        Pipeline.new()
        |> Pipeline.add(:l1, &Penalties.l1/2, weight: 0.1)

      tensor = Nx.tensor([1.0, -2.0, 3.0])
      {total, _metrics} = Pipeline.compute(pipeline, tensor)

      # L1 = 6, weighted = 0.1 * 6 = 0.6
      assert_close(total, Nx.tensor(0.6))
    end

    test "combines multiple penalties" do
      pipeline =
        Pipeline.new()
        |> Pipeline.add(:l1, &Penalties.l1/2, weight: 0.1)
        |> Pipeline.add(:l2, &Penalties.l2/2, weight: 0.01)

      tensor = Nx.tensor([1.0, 2.0])
      {total, metrics} = Pipeline.compute(pipeline, tensor)

      # L1 = 3, weighted = 0.3
      # L2 = 5, weighted = 0.05
      # Total = 0.35
      assert_close(total, Nx.tensor(0.35))
      assert Map.has_key?(metrics, "l1")
      assert Map.has_key?(metrics, "l2")
    end

    test "skips disabled penalties" do
      pipeline =
        Pipeline.new()
        |> Pipeline.add(:l1, &Penalties.l1/2, weight: 0.1)
        |> Pipeline.add(:l2, &Penalties.l2/2, weight: 0.01)
        |> Pipeline.set_enabled(:l2, false)

      tensor = Nx.tensor([1.0, 2.0])
      {total, metrics} = Pipeline.compute(pipeline, tensor)

      # Only L1: 3 * 0.1 = 0.3
      assert_close(total, Nx.tensor(0.3))
      refute Map.has_key?(metrics, "l2")
    end

    test "returns zero for empty pipeline" do
      pipeline = Pipeline.new()
      tensor = Nx.tensor([1.0, 2.0, 3.0])
      {total, metrics} = Pipeline.compute(pipeline, tensor)

      assert_close(total, Nx.tensor(0.0))
      assert metrics == %{}
    end

    test "metrics contain raw and weighted values" do
      pipeline =
        Pipeline.new()
        |> Pipeline.add(:l1, &Penalties.l1/2, weight: 0.1)

      tensor = Nx.tensor([1.0, -2.0, 3.0])
      {_total, metrics} = Pipeline.compute(pipeline, tensor)

      assert_in_delta metrics["l1"], 6.0, 1.0e-5
      assert_in_delta metrics["l1_weighted"], 0.6, 1.0e-5
    end

    test "metrics contain total" do
      pipeline =
        Pipeline.new()
        |> Pipeline.add(:l1, &Penalties.l1/2, weight: 0.1)

      tensor = Nx.tensor([1.0, -2.0, 3.0])
      {_total, metrics} = Pipeline.compute(pipeline, tensor)

      assert_in_delta metrics["total"], 0.6, 1.0e-5
    end

    test "extra_args option merges into penalty opts" do
      pipeline =
        Pipeline.new()
        |> Pipeline.add(:l1, &Penalties.l1/2, weight: 1.0, opts: [lambda: 0.5])

      tensor = Nx.tensor([1.0, -2.0, 3.0])

      # With extra_args overriding lambda
      {total, _} = Pipeline.compute(pipeline, tensor, extra_args: [lambda: 2.0])

      # L1 with lambda=2.0: 2.0 * 6 = 12.0
      assert_close(total, Nx.tensor(12.0))
    end

    test "works with divergence functions" do
      pipeline =
        Pipeline.new()
        |> Pipeline.add(:entropy, &Divergences.entropy/2, weight: 0.1, opts: [mode: :penalty])

      logprobs = Nx.log(Nx.tensor([0.25, 0.25, 0.25, 0.25]))
      {total, _metrics} = Pipeline.compute(pipeline, logprobs)

      assert_finite(total)
    end
  end

  describe "gradient flow" do
    test "gradient flows through pipeline" do
      pipeline =
        Pipeline.new()
        |> Pipeline.add(:l2, &Penalties.l2/2, weight: 0.01)

      grad_fn =
        Nx.Defn.grad(fn tensor ->
          Pipeline.compute_total(pipeline, tensor)
        end)

      tensor = Nx.tensor([1.0, 2.0, 3.0])
      grads = grad_fn.(tensor)

      assert_finite(grads)
      # L2 gradient: 2 * weight * x = 0.02 * [1, 2, 3]
      assert_close(grads, Nx.tensor([0.02, 0.04, 0.06]))
    end
  end

  describe "compute_total/3" do
    test "returns only total without metrics" do
      pipeline =
        Pipeline.new()
        |> Pipeline.add(:l1, &Penalties.l1/2, weight: 0.1)

      tensor = Nx.tensor([1.0, -2.0, 3.0])
      total = Pipeline.compute_total(pipeline, tensor)

      assert_scalar(total)
      assert_close(total, Nx.tensor(0.6))
    end
  end
end
