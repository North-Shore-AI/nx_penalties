defmodule NxPenalties.TestHelpers do
  @moduledoc """
  Common test utilities for NxPenalties.
  """

  import ExUnit.Assertions

  @doc """
  Assert two tensors are approximately equal.

  ## Options

    * `:atol` - Absolute tolerance. Default: `1.0e-5`
    * `:rtol` - Relative tolerance. Default: `1.0e-5`

  Uses the formula: |a - b| <= atol + rtol * |b|
  """
  def assert_close(actual, expected, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-5)
    rtol = Keyword.get(opts, :rtol, 1.0e-5)

    actual_data = Nx.to_flat_list(actual)
    expected_data = Nx.to_flat_list(expected)

    assert length(actual_data) == length(expected_data),
           "Shape mismatch: #{inspect(Nx.shape(actual))} vs #{inspect(Nx.shape(expected))}"

    Enum.zip(actual_data, expected_data)
    |> Enum.each(fn {a, e} ->
      diff = abs(a - e)
      tolerance = atol + rtol * abs(e)

      assert diff <= tolerance,
             """
             Values not close:
               actual:   #{a}
               expected: #{e}
               diff:     #{diff}
               tolerance: #{tolerance} (atol=#{atol}, rtol=#{rtol})
             """
    end)
  end

  @doc """
  Assert tensor contains no NaN values.
  """
  def assert_no_nan(tensor) do
    has_nan = tensor |> Nx.is_nan() |> Nx.any() |> Nx.to_number()
    assert has_nan == 0, "Tensor contains NaN values"
  end

  @doc """
  Assert tensor contains no Inf values.
  """
  def assert_no_inf(tensor) do
    has_inf = tensor |> Nx.is_infinity() |> Nx.any() |> Nx.to_number()
    assert has_inf == 0, "Tensor contains Inf values"
  end

  @doc """
  Assert tensor is finite (no NaN or Inf).
  """
  def assert_finite(tensor) do
    assert_no_nan(tensor)
    assert_no_inf(tensor)
  end

  @doc """
  Assert tensor is a valid scalar.
  """
  def assert_scalar(tensor) do
    assert Nx.shape(tensor) == {}, "Expected scalar, got shape #{inspect(Nx.shape(tensor))}"
    assert_finite(tensor)
  end

  @doc """
  Assert function can be JIT compiled without error.
  """
  def assert_jit_compiles(fun, args) do
    jit_fn = Nx.Defn.jit(fun)
    result = apply(jit_fn, args)
    assert_finite(result)
    result
  end

  @doc """
  Generate random tensor for testing.
  """
  def random_tensor(shape, opts \\ []) do
    type = Keyword.get(opts, :type, :f32)
    min = Keyword.get(opts, :min, -1.0)
    max = Keyword.get(opts, :max, 1.0)

    key = Nx.Random.key(System.unique_integer())
    {tensor, _new_key} = Nx.Random.uniform(key, min, max, shape: shape, type: type)
    tensor
  end

  @doc """
  Generate random log-probability tensor (normalized).
  """
  def random_logprobs(shape) do
    raw = random_tensor(shape)
    # Normalize so exp sums to 1
    Nx.subtract(raw, Nx.logsumexp(raw, axes: [-1], keep_axes: true))
  end
end
