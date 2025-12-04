# 10: Backend Compatibility Implementation Specification

## Overview

NxPenalties must work correctly across all Nx backends. This document specifies backend considerations, testing strategies, and compatibility requirements.

## Supported Backends

| Backend | Platform | Priority | Notes |
|---------|----------|----------|-------|
| `Nx.BinaryBackend` | CPU (default) | Critical | Reference implementation, always available |
| `EXLA.Backend` | CPU/GPU/TPU | High | XLA compilation, production use |
| `Torchx.Backend` | CPU/GPU | Medium | PyTorch interop |

## Backend Abstraction Principles

### Rule 1: Pure Nx Operations Only

```elixir
# Good: Uses only Nx functions
defn l1(tensor, opts) do
  lambda = opts[:lambda] || 0.01
  Nx.sum(Nx.abs(tensor)) |> Nx.multiply(lambda)
end

# Bad: Uses backend-specific operations
defn l1_bad(tensor, opts) do
  # Don't do this!
  EXLA.some_function(tensor)
end
```

### Rule 2: No Backend Detection in Hot Path

```elixir
# Bad: Runtime backend check
defn compute(tensor, opts) do
  if Nx.backend(tensor) == EXLA.Backend do
    # Optimized path
  else
    # Fallback
  end
end

# Good: Same code path for all backends
defn compute(tensor, opts) do
  # Single implementation using Nx primitives
end
```

### Rule 3: Backend Selection is User's Responsibility

```elixir
# User code sets backend, not library
Nx.default_backend(EXLA.Backend)

# Or per-tensor
tensor = Nx.tensor([1.0, 2.0], backend: EXLA.Backend)

# Library code is backend-agnostic
NxPenalties.l1(tensor)  # Works on any backend
```

---

## Nx Operations Compatibility

### Universally Supported Operations

These operations work identically across all backends:

| Category | Operations |
|----------|-----------|
| Arithmetic | `add`, `subtract`, `multiply`, `divide`, `power` |
| Reduction | `sum`, `mean`, `reduce_max`, `reduce_min` |
| Comparison | `greater`, `less`, `equal`, `select` |
| Unary | `abs`, `negate`, `sign`, `sqrt`, `exp`, `log` |
| Shape | `reshape`, `transpose`, `broadcast` |
| Linear Algebra | `dot` |

### Operations with Potential Differences

| Operation | Issue | Mitigation |
|-----------|-------|------------|
| `Nx.logsumexp` | Numerical precision varies | Trust Nx's implementation |
| `Nx.Random.*` | Different RNG implementations | Use explicit keys |
| `Nx.LinAlg.svd` | Not all backends support | Don't use, or feature-flag |
| Custom `Nx.Defn.Kernel.hook` | Side-effects, not JIT-friendly | Use sparingly |

### Type Support

| Type | Binary | EXLA | Torchx |
|------|--------|------|--------|
| f32 | Yes | Yes | Yes |
| f64 | Yes | Yes | Yes |
| f16 | No* | Yes | Yes |
| bf16 | No* | Yes | Yes |
| s32/s64 | Yes | Yes | Yes |

*Binary backend doesn't support f16/bf16 natively; will upcast.

**Implication**: Test with f32 by default; add specific f16/bf16 tests for GPU backends.

---

## Testing Strategy

### Test Matrix

```
        │ Binary │ EXLA CPU │ EXLA GPU │ Torchx │
────────┼────────┼──────────┼──────────┼────────┤
 l1     │   ✓    │    ✓     │    ✓     │   ✓    │
 l2     │   ✓    │    ✓     │    ✓     │   ✓    │
 elastic│   ✓    │    ✓     │    ✓     │   ✓    │
 kl_div │   ✓    │    ✓     │    ✓     │   ✓    │
 entropy│   ✓    │    ✓     │    ✓     │   ✓    │
 ...    │   ✓    │    ✓     │    ✓     │   ✓    │
```

### CI Configuration

```yaml
# .github/workflows/test.yml
jobs:
  test-binary:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: erlef/setup-beam@v1
        with:
          elixir-version: "1.16"
          otp-version: "26"
      - run: mix deps.get
      - run: mix test

  test-exla-cpu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: erlef/setup-beam@v1
        with:
          elixir-version: "1.16"
          otp-version: "26"
      - run: mix deps.get
      - run: MIX_ENV=test mix deps.compile exla
      - run: NX_BACKEND=exla mix test --only exla

  test-exla-gpu:
    runs-on: [self-hosted, gpu]
    steps:
      # Similar, with CUDA setup
```

### Backend Test Helper

```elixir
# test/support/backend_helpers.ex
defmodule NxPenalties.BackendHelpers do
  @available_backends [
    {Nx.BinaryBackend, []},
    # EXLA and Torchx added dynamically
  ]

  def available_backends do
    backends = @available_backends

    backends =
      if Code.ensure_loaded?(EXLA.Backend) do
        [{EXLA.Backend, []} | backends]
      else
        backends
      end

    if Code.ensure_loaded?(Torchx.Backend) do
      [{Torchx.Backend, []} | backends]
    else
      backends
    end
  end

  def for_each_backend(fun) do
    original = Nx.default_backend()

    try do
      for {backend, opts} <- available_backends() do
        Nx.default_backend({backend, opts})
        fun.(backend)
      end
    after
      Nx.default_backend(original)
    end
  end

  defmacro test_on_all_backends(name, do: block) do
    quote do
      for {backend, _} <- NxPenalties.BackendHelpers.available_backends() do
        @tag backend: backend
        test "#{unquote(name)} [#{inspect(backend)}]" do
          Nx.default_backend(backend)
          unquote(block)
        end
      end
    end
  end
end
```

### Backend-Specific Test

```elixir
defmodule NxPenalties.BackendTest do
  use ExUnit.Case, async: true

  import NxPenalties.BackendHelpers
  import NxPenalties.TestHelpers

  describe "l1 across backends" do
    test_on_all_backends "produces consistent results" do
      tensor = Nx.tensor([1.0, -2.0, 3.0])
      result = NxPenalties.l1(tensor, lambda: 0.1)

      assert_close(result, Nx.tensor(0.6))
    end
  end

  describe "kl_divergence across backends" do
    test_on_all_backends "identical distributions have zero KL" do
      logprobs = random_logprobs({1, 10})
      result = NxPenalties.kl_divergence(logprobs, logprobs)

      assert_close(result, Nx.tensor(0.0), atol: 1.0e-5)
    end
  end

  # EXLA-specific tests
  @tag :exla
  @tag skip: !Code.ensure_loaded?(EXLA.Backend)
  describe "EXLA-specific" do
    test "handles device placement" do
      Nx.default_backend(EXLA.Backend)

      tensor = Nx.tensor([1.0, 2.0])
      result = NxPenalties.l1(tensor)

      assert Nx.backend(result) == EXLA.Backend
    end

    test "JIT compilation works" do
      Nx.default_backend(EXLA.Backend)

      jit_l1 = Nx.Defn.jit(&NxPenalties.Penalties.l1/2)
      tensor = Nx.tensor([1.0, 2.0])

      # First call compiles
      result1 = jit_l1.(tensor, [lambda: 0.1])

      # Second call uses cached
      result2 = jit_l1.(tensor, [lambda: 0.1])

      assert_close(result1, result2)
    end
  end
end
```

---

## JIT Compilation Requirements

### All Defn Functions Must:

1. **Use only Nx operations**
2. **Avoid Elixir control flow on tensor values**
3. **Have compile-time-known shapes** (or use dynamic shapes carefully)
4. **Not call non-defn functions** (except via hooks)

### Shape Handling

```elixir
# Good: Shape operations inside defn
defn reshape_and_compute(tensor, opts) do
  shape = Nx.shape(tensor)  # OK - shape is compile-time
  # ...
end

# Bad: Dynamic shape from runtime
def compute_bad(tensor, target_shape) do
  # This won't work if target_shape is runtime-dynamic
  reshaped = Nx.reshape(tensor, target_shape)
  # ...
end
```

### Options Pattern

```elixir
# Options must be compile-time constants for fusion
defn l1(tensor, opts \\ []) do
  lambda = opts[:lambda] || 0.01  # Resolved at compile time
  # ...
end

# If lambda needs to be runtime tensor:
defn l1_dynamic(tensor, lambda) do
  Nx.multiply(Nx.sum(Nx.abs(tensor)), lambda)
end
```

---

## Performance Considerations

### Memory Layout

Different backends may have different optimal memory layouts:

```elixir
# Let backend choose optimal layout
tensor = Nx.tensor([[1, 2], [3, 4]])

# Don't assume row-major or column-major
# Use Nx operations that abstract layout
```

### Device Transfer

```elixir
# Minimize CPU <-> GPU transfers
# Keep tensors on device throughout computation

# Bad: Repeated transfers
for _ <- 1..1000 do
  tensor = Nx.tensor([1, 2, 3])  # Transfer each iteration
  result = NxPenalties.l1(tensor)
  Nx.to_number(result)  # Transfer back
end

# Good: Batch operations
tensor = Nx.tensor([1, 2, 3])  # One transfer
results = for _ <- 1..1000 do
  NxPenalties.l1(tensor)
end
# One transfer back at end
```

### Compilation Caching

EXLA caches compiled functions. Ensure consistent signatures for cache hits:

```elixir
# Same function with same shapes reuses cached compilation
tensor1 = Nx.random_uniform({100})
tensor2 = Nx.random_uniform({100})

result1 = NxPenalties.l1(tensor1)  # Compiles
result2 = NxPenalties.l1(tensor2)  # Uses cache

# Different shape triggers recompilation
tensor3 = Nx.random_uniform({200})
result3 = NxPenalties.l1(tensor3)  # Recompiles
```

---

## Fallback Patterns

### Feature Detection

```elixir
defmodule NxPenalties.Features do
  def supports_f16? do
    case Nx.default_backend() do
      {EXLA.Backend, _} -> true
      {Torchx.Backend, _} -> true
      _ -> false
    end
  end

  def supports_svd? do
    # Check if SVD is available
    Code.ensure_loaded?(Nx.LinAlg) and
      function_exported?(Nx.LinAlg, :svd, 1)
  end
end
```

### Graceful Degradation

```elixir
def orthogonality(tensor, opts) do
  mode = opts[:mode] || :soft

  case mode do
    :spectral ->
      if NxPenalties.Features.supports_svd?() do
        spectral_orthogonality(tensor)
      else
        Logger.warning("SVD not available, falling back to :soft mode")
        soft_orthogonality(tensor)
      end

    _ ->
      soft_orthogonality(tensor)
  end
end
```

---

## Documentation Requirements

### Per-Function Documentation

```elixir
@doc """
L1 penalty (Lasso regularization).

## Backend Compatibility

Works on all Nx backends. Tested on:
- Nx.BinaryBackend (reference)
- EXLA.Backend (CPU and GPU)
- Torchx.Backend

## Performance Notes

On GPU (EXLA/Torchx), large tensors (>1M elements) benefit
significantly from parallel reduction.
"""
```

### README Section

```markdown
## Backend Support

NxPenalties works with any Nx backend:

```elixir
# Default (BinaryBackend)
NxPenalties.l1(tensor)

# EXLA (recommended for GPU)
Nx.default_backend(EXLA.Backend)
NxPenalties.l1(tensor)

# Torchx
Nx.default_backend(Torchx.Backend)
NxPenalties.l1(tensor)
```

All penalty functions produce identical results across backends
(within floating-point tolerance).
```

---

## Checklist

- [ ] All functions use only Nx operations
- [ ] No backend-specific code in core library
- [ ] Tests pass on BinaryBackend
- [ ] Tests pass on EXLA (CPU)
- [ ] Tests pass on EXLA (GPU) - if available
- [ ] Tests pass on Torchx - if available
- [ ] JIT compilation tests for all functions
- [ ] Numerical consistency tests across backends
- [ ] Documentation specifies backend compatibility
- [ ] CI runs tests on multiple backends
