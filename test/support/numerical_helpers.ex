defmodule NxPenalties.NumericalHelpers do
  @moduledoc """
  Numerical testing utilities including gradient verification.
  """

  import ExUnit.Assertions
  import NxPenalties.TestHelpers

  @doc """
  Verify gradients using finite differences.

  Computes numerical gradient and compares to autodiff gradient.
  Uses central differences: (f(x+h) - f(x-h)) / 2h
  """
  def verify_gradients(fun, input, opts \\ []) do
    epsilon = Keyword.get(opts, :epsilon, 1.0e-4)
    tolerance = Keyword.get(opts, :tolerance, 1.0e-3)

    # Autodiff gradient
    grad_fn = Nx.Defn.grad(fun)
    autodiff_grad = grad_fn.(input)

    # Numerical gradient (finite differences)
    numerical_grad = compute_numerical_gradient(fun, input, epsilon)

    # Compare
    diff = Nx.subtract(autodiff_grad, numerical_grad)
    max_diff = diff |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

    assert max_diff < tolerance,
           """
           Gradient mismatch:
             max difference: #{max_diff}
             tolerance: #{tolerance}
             autodiff: #{inspect(Nx.to_flat_list(autodiff_grad))}
             numerical: #{inspect(Nx.to_flat_list(numerical_grad))}
           """
  end

  defp compute_numerical_gradient(fun, input, epsilon) do
    flat_input = Nx.to_flat_list(input)
    shape = Nx.shape(input)
    type = Nx.type(input)

    grads =
      flat_input
      |> Enum.with_index()
      |> Enum.map(fn {_, i} ->
        # f(x + epsilon)
        plus = perturb_at_index(flat_input, i, epsilon)
        plus_tensor = Nx.tensor(plus, type: type) |> Nx.reshape(shape)
        f_plus = fun.(plus_tensor) |> Nx.to_number()

        # f(x - epsilon)
        minus = perturb_at_index(flat_input, i, -epsilon)
        minus_tensor = Nx.tensor(minus, type: type) |> Nx.reshape(shape)
        f_minus = fun.(minus_tensor) |> Nx.to_number()

        # Central difference
        (f_plus - f_minus) / (2 * epsilon)
      end)

    Nx.tensor(grads, type: type) |> Nx.reshape(shape)
  end

  defp perturb_at_index(list, index, delta) do
    List.update_at(list, index, &(&1 + delta))
  end

  @doc """
  Check numerical stability across value ranges.
  """
  def check_stability(fun, ranges, opts \\ []) do
    samples = Keyword.get(opts, :samples, 100)

    Enum.each(ranges, fn {min, max, _description} ->
      for _ <- 1..samples do
        input = random_tensor({10}, min: min, max: max)
        result = fun.(input)

        assert_finite(result)
      end
    end)
  end
end
