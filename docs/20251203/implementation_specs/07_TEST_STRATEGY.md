# 07: Test Strategy Implementation Specification

## Overview

This document defines the testing strategy for NxPenalties, drawing on principles from Supertester and adapted for numerical/ML library testing. The goal is deterministic, comprehensive, fast tests that catch both correctness and numerical issues.

## Testing Principles (from Supertester)

1. **Zero Sleep** - No timing-based synchronization; use proper patterns
2. **Parallel Execution** - All tests run with `async: true`
3. **Isolation** - No shared state between tests
4. **Deterministic** - Same input → same output, always
5. **Expressive Assertions** - Domain-specific helpers

## Numerical Testing Challenges

| Challenge | Solution |
|-----------|----------|
| Floating point comparison | `assert_close/3` with configurable tolerance |
| NaN/Inf detection | Explicit checks in tests |
| Backend differences | Test across multiple backends |
| Gradient correctness | Finite difference verification |
| JIT compilation | Explicit JIT tests |

## Test Organization

```
test/
├── nx_penalties/
│   ├── penalties_test.exs       # L1, L2, Elastic Net
│   ├── divergences_test.exs     # KL, JS, Entropy
│   ├── constraints_test.exs     # Orthogonality, etc.
│   ├── pipeline_test.exs        # Composition
│   └── integration/
│       ├── axon_test.exs        # Axon helpers
│       └── polaris_test.exs     # Gradient transforms
├── support/
│   ├── test_helpers.ex          # Common utilities
│   ├── numerical_helpers.ex     # Numerical assertions
│   └── reference_implementations.ex  # Python parity checks
└── property/
    ├── penalties_property_test.exs  # Property-based tests
    └── generators.ex            # StreamData generators
```

## Test Helpers Module

### File: test/support/test_helpers.ex

```elixir
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

    Nx.random_uniform(shape, min, max, type: type)
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
```

## Numerical Helpers Module

### File: test/support/numerical_helpers.ex

```elixir
defmodule NxPenalties.NumericalHelpers do
  @moduledoc """
  Numerical testing utilities including gradient verification.
  """

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

    Enum.each(ranges, fn {min, max, description} ->
      for _ <- 1..samples do
        input = Nx.random_uniform({10}, min, max)
        result = fun.(input)

        assert_finite(result),
          "Function produced non-finite result for range #{description} (#{min}, #{max})"
      end
    end)
  end
end
```

## Backend Testing

### File: test/support/backend_helpers.ex

```elixir
defmodule NxPenalties.BackendHelpers do
  @moduledoc """
  Helpers for testing across multiple Nx backends.
  """

  @doc """
  Get list of available backends for testing.
  """
  def available_backends do
    backends = [{Nx.BinaryBackend, []}]

    # Check for EXLA
    backends =
      if Code.ensure_loaded?(EXLA.Backend) do
        [{EXLA.Backend, []} | backends]
      else
        backends
      end

    # Check for Torchx
    if Code.ensure_loaded?(Torchx.Backend) do
      [{Torchx.Backend, []} | backends]
    else
      backends
    end
  end

  @doc """
  Run test function on all available backends.
  """
  def with_backends(fun) do
    for {backend, opts} <- available_backends() do
      Nx.default_backend({backend, opts})
      try do
        fun.(backend)
      after
        Nx.default_backend(Nx.BinaryBackend)
      end
    end
  end

  @doc """
  Tag for tests that require EXLA.
  """
  def exla_tag do
    if Code.ensure_loaded?(EXLA.Backend), do: [], else: [:skip]
  end
end
```

## Test Case Templates

### Basic Penalty Test

```elixir
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

    test "returns zero for zero tensor" do
      tensor = Nx.tensor([0.0, 0.0, 0.0])
      result = Penalties.l1(tensor, lambda: 1.0)
      assert_close(result, Nx.tensor(0.0))
    end

    test "correct value for known input" do
      tensor = Nx.tensor([1.0, -2.0, 3.0])
      result = Penalties.l1(tensor, lambda: 0.1)
      # |1| + |-2| + |3| = 6, * 0.1 = 0.6
      assert_close(result, Nx.tensor(0.6))
    end

    test "gradient is sign function" do
      verify_gradients(
        fn x -> Penalties.l1(x, lambda: 1.0) end,
        Nx.tensor([2.0, -3.0, 0.5])
      )
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
          {-1.0e6, 1.0e6, "large"},
        ]
      )
    end
  end
end
```

### Divergence Test

```elixir
defmodule NxPenalties.DivergencesTest do
  use ExUnit.Case, async: true

  import NxPenalties.TestHelpers
  import NxPenalties.NumericalHelpers

  alias NxPenalties.Divergences

  describe "kl_divergence/3" do
    test "zero for identical distributions" do
      logprobs = random_logprobs({1, 10})
      result = Divergences.kl_divergence(logprobs, logprobs)
      assert_close(result, Nx.tensor(0.0), atol: 1.0e-6)
    end

    test "non-negative" do
      p = random_logprobs({5, 20})
      q = random_logprobs({5, 20})
      result = Divergences.kl_divergence(p, q)
      assert Nx.to_number(result) >= -1.0e-6  # Allow tiny numerical error
    end

    test "asymmetric: KL(P||Q) != KL(Q||P)" do
      p = random_logprobs({1, 10})
      q = random_logprobs({1, 10})

      kl_pq = Divergences.kl_divergence(p, q)
      kl_qp = Divergences.kl_divergence(q, p)

      # Should generally be different (unless p == q)
      # This is a probabilistic test
      refute_close(kl_pq, kl_qp)
    end

    test "handles batch dimension" do
      p = random_logprobs({4, 10})
      q = random_logprobs({4, 10})

      none = Divergences.kl_divergence(p, q, reduction: :none)
      assert Nx.shape(none) == {4}

      mean = Divergences.kl_divergence(p, q, reduction: :mean)
      assert_scalar(mean)
    end

    test "gradient is finite" do
      p = random_logprobs({2, 8})
      q = random_logprobs({2, 8})

      grad_fn = Nx.Defn.grad(fn x ->
        Divergences.kl_divergence(x, q)
      end)

      grads = grad_fn.(p)
      assert_finite(grads)
    end
  end

  # Helper for refuting closeness
  defp refute_close(a, b) do
    diff = Nx.subtract(a, b) |> Nx.abs() |> Nx.to_number()
    assert diff > 1.0e-6, "Expected values to differ, but got diff=#{diff}"
  end
end
```

### Property-Based Tests

```elixir
defmodule NxPenalties.PenaltiesPropertyTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  import NxPenalties.TestHelpers

  alias NxPenalties.Penalties

  property "l1 is always non-negative" do
    check all tensor <- tensor_generator() do
      result = Penalties.l1(tensor, lambda: 1.0)
      assert Nx.to_number(result) >= 0
    end
  end

  property "l2 is always non-negative" do
    check all tensor <- tensor_generator() do
      result = Penalties.l2(tensor, lambda: 1.0)
      assert Nx.to_number(result) >= 0
    end
  end

  property "elastic_net(alpha=1) == l1" do
    check all tensor <- tensor_generator() do
      l1 = Penalties.l1(tensor, lambda: 0.1)
      elastic = Penalties.elastic_net(tensor, lambda: 0.1, l1_ratio: 1.0)
      assert_close(l1, elastic, atol: 1.0e-5)
    end
  end

  property "elastic_net(alpha=0) == l2" do
    check all tensor <- tensor_generator() do
      l2 = Penalties.l2(tensor, lambda: 0.1)
      elastic = Penalties.elastic_net(tensor, lambda: 0.1, l1_ratio: 0.0)
      assert_close(l2, elastic, atol: 1.0e-5)
    end
  end

  # Generators
  defp tensor_generator do
    gen all shape <- shape_generator(),
            values <- list_of(float(min: -100, max: 100), length: Enum.product(Tuple.to_list(shape))) do
      Nx.tensor(values) |> Nx.reshape(shape)
    end
  end

  defp shape_generator do
    one_of([
      constant({4}),
      constant({2, 3}),
      constant({2, 3, 4}),
      tuple({integer(1..5), integer(1..5)}),
    ])
  end
end
```

## Integration Tests

### Axon Integration Test

```elixir
defmodule NxPenalties.Integration.AxonTest do
  use ExUnit.Case, async: true

  import NxPenalties.TestHelpers

  alias NxPenalties.Integration.Axon, as: AxonIntegration

  describe "wrap_loss/3" do
    test "adds penalty to base loss" do
      base_loss = fn _y_true, y_pred -> Nx.mean(Nx.power(y_pred, 2)) end
      wrapped = AxonIntegration.wrap_loss(
        base_loss,
        &NxPenalties.Penalties.l2/2,
        lambda: 0.1
      )

      y_true = Nx.tensor([0.0])
      y_pred = Nx.tensor([2.0])

      base_value = base_loss.(y_true, y_pred) |> Nx.to_number()
      wrapped_value = wrapped.(y_true, y_pred) |> Nx.to_number()

      assert wrapped_value > base_value
    end
  end

  describe "capture_activation/2" do
    @tag :integration
    test "captures intermediate values" do
      model =
        Axon.input("x", shape: {nil, 4})
        |> Axon.dense(8)
        |> AxonIntegration.capture_activation(:hidden)
        |> Axon.dense(2)

      {init_fn, predict_fn} = Axon.build(model, mode: :train)
      params = init_fn.(Nx.template({1, 4}, :f32), %{})

      input = random_tensor({1, 4})
      {_output, state} = predict_fn.(params, input)

      captures = AxonIntegration.extract_captures(state)

      assert Map.has_key?(captures, :hidden)
      assert Nx.shape(captures.hidden) == {1, 8}
    end
  end
end
```

## Test Configuration

### test/test_helper.exs

```elixir
# Configure ExUnit
ExUnit.start(exclude: [:skip])

# Compile test support modules
Code.require_file("support/test_helpers.ex", __DIR__)
Code.require_file("support/numerical_helpers.ex", __DIR__)
Code.require_file("support/backend_helpers.ex", __DIR__)

# Set default backend for tests
Nx.default_backend(Nx.BinaryBackend)

# Optional: Configure EXLA for GPU tests
if Code.ensure_loaded?(EXLA) do
  Application.put_env(:exla, :clients,
    cuda: [platform: :cuda],
    rocm: [platform: :rocm],
    tpu: [platform: :tpu],
    host: [platform: :host]
  )
end
```

## CI Configuration

### .github/workflows/ci.yml

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        elixir: ['1.15', '1.16']
        otp: ['25', '26']
        backend: ['binary', 'exla']

    steps:
      - uses: actions/checkout@v4

      - uses: erlef/setup-beam@v1
        with:
          elixir-version: ${{ matrix.elixir }}
          otp-version: ${{ matrix.otp }}

      - name: Install dependencies
        run: mix deps.get

      - name: Compile
        run: mix compile --warnings-as-errors

      - name: Run tests (Binary Backend)
        if: matrix.backend == 'binary'
        run: mix test

      - name: Run tests (EXLA Backend)
        if: matrix.backend == 'exla'
        run: |
          MIX_ENV=test mix deps.compile exla
          NX_BACKEND=exla mix test

      - name: Check formatting
        run: mix format --check-formatted

      - name: Run Credo
        run: mix credo --strict
```

## Coverage Requirements

| Module | Target Coverage |
|--------|-----------------|
| `NxPenalties.Penalties` | 100% |
| `NxPenalties.Divergences` | 100% |
| `NxPenalties.Constraints` | 95% |
| `NxPenalties.Pipeline` | 95% |
| `NxPenalties.Integration.*` | 90% |

## Test Checklist

- [ ] All penalty functions have unit tests
- [ ] Gradient verification for all differentiable functions
- [ ] Numerical stability tests for edge cases
- [ ] JIT compilation tests
- [ ] Property-based tests for invariants
- [ ] Multi-backend tests (at least Binary + EXLA)
- [ ] Integration tests with real Axon models
- [ ] Pipeline composition tests
- [ ] Error case tests (invalid inputs)
- [ ] Documentation tests (doctests)
