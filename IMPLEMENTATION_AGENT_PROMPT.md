# NxPenalties Full Implementation Agent Prompt

**Project**: NxPenalties - Composable regularization penalties for the Nx ecosystem
**Target**: v0.1.0 MVP
**Approach**: Test-Driven Development (TDD) with Supertester patterns

---

## Executive Summary

You are implementing **NxPenalties**, a standalone Elixir library providing composable regularization penalties for ML training with Nx. This fills a gap between Axon (model graph) and Polaris (optimization) by providing loss-based regularization infrastructure.

**Critical Requirements:**
1. All tests passing (`mix test` - 0 failures)
2. No compiler warnings (`mix compile --warnings-as-errors`)
3. No Credo issues (`mix credo --strict`)
4. Properly formatted (`mix format`)
5. Working examples that demonstrate all features

---

## Required Reading (In Order)

Before implementing, you MUST read these specification documents in full. They contain the complete API design, implementation details, and test cases:

### Core Architecture
```
docs/20251203/implementation_specs/README.md                 # Project overview, phases, module structure
docs/20251203/implementation_specs/00_ARCHITECTURE.md        # Design philosophy, module responsibilities
docs/20251203/implementation_specs/08_API_REFERENCE.md       # Complete API with @spec and options
docs/20251203/implementation_specs/09_NUMERICAL_STABILITY.md # Edge cases, NaN/Inf handling
docs/20251203/implementation_specs/10_BACKEND_COMPATIBILITY.md # EXLA/Torchx testing
```

### Phase 1 Implementations (v0.1)
```
docs/20251203/implementation_specs/01_PENALTY_PRIMITIVES.md  # L1, L2, Elastic Net
docs/20251203/implementation_specs/02_DIVERGENCES.md        # KL, JS, Entropy
docs/20251203/implementation_specs/04_PIPELINE.md           # Composition engine
docs/20251203/implementation_specs/05_AXON_INTEGRATION.md   # Axon training helpers
docs/20251203/implementation_specs/07_TEST_STRATEGY.md      # TDD approach, helpers
```

### Phase 2 Implementations (v0.2 - stub only)
```
docs/20251203/implementation_specs/03_CONSTRAINTS.md         # Orthogonality, Consistency
docs/20251203/implementation_specs/06_POLARIS_INTEGRATION.md # Gradient transforms
docs/20251203/implementation_specs/11_GRADIENT_TRACKING.md   # Gradient norm monitoring
```

### ADRs (Architecture Decision Records)
```
docs/20251203/implementation_specs/regularizer_adrs/README.md
docs/20251203/implementation_specs/regularizer_adrs/adrs/ADR-001_l1_sparsity.md
docs/20251203/implementation_specs/regularizer_adrs/adrs/ADR-002_l2_weight_decay.md
docs/20251203/implementation_specs/regularizer_adrs/adrs/ADR-003_elastic_net.md
docs/20251203/implementation_specs/regularizer_adrs/adrs/ADR-004_kl_divergence.md
docs/20251203/implementation_specs/regularizer_adrs/adrs/ADR-005_entropy.md
docs/20251203/implementation_specs/regularizer_adrs/adrs/ADR-006_consistency.md
docs/20251203/implementation_specs/regularizer_adrs/adrs/ADR-008_orthogonality.md
docs/20251203/implementation_specs/regularizer_adrs/adrs/ADR-009_gradient_tracking.md
```

---

## Project Setup

### 1. Initialize Mix Project
```bash
cd /home/home/p/g/North-Shore-AI/nx_penalties
mix new . --app nx_penalties --module NxPenalties
```

### 2. Configure mix.exs
```elixir
defmodule NxPenalties.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/North-Shore-AI/nx_penalties"

  def project do
    [
      app: :nx_penalties,
      version: @version,
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps(),

      # Docs
      name: "NxPenalties",
      source_url: @source_url,
      docs: docs(),

      # Test
      test_coverage: [tool: ExCoveralls],
      preferred_cli_env: [
        coveralls: :test,
        "coveralls.html": :test
      ],

      # Dialyzer
      dialyzer: [
        plt_file: {:no_warn, "priv/plts/dialyzer.plt"},
        plt_add_apps: [:ex_unit]
      ],

      # Aliases
      aliases: aliases()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      # Core
      {:nx, "~> 0.9"},
      {:nimble_options, "~> 1.0"},
      {:telemetry, "~> 1.0"},

      # Optional integrations
      {:axon, "~> 0.6", optional: true},
      {:polaris, "~> 0.1", optional: true},

      # Test
      {:exla, "~> 0.9", only: :test},
      {:stream_data, "~> 1.0", only: [:test, :dev]},
      {:excoveralls, "~> 0.18", only: :test},

      # Dev
      {:ex_doc, "~> 0.34", only: :dev, runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false}
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: ["README.md", "CHANGELOG.md"],
      source_ref: "v#{@version}"
    ]
  end

  defp aliases do
    [
      quality: ["format", "credo --strict", "dialyzer"]
    ]
  end
end
```

### 3. Create Directory Structure
```
lib/
├── nx_penalties.ex                    # Main entry module
└── nx_penalties/
    ├── penalties.ex                   # L1, L2, Elastic Net
    ├── penalties/
    │   └── validation.ex              # NimbleOptions schemas
    ├── divergences.ex                 # KL, JS, Entropy
    ├── divergences/
    │   └── validation.ex
    ├── constraints.ex                 # Orthogonality, Consistency (v0.2 stubs)
    ├── pipeline.ex                    # Composition engine
    ├── gradient_tracker.ex            # Gradient norm monitoring (v0.2 stub)
    ├── telemetry.ex                   # Instrumentation
    └── integration/
        ├── axon.ex                    # Axon training helpers
        └── polaris.ex                 # Gradient transforms (v0.2 stub)

test/
├── test_helper.exs
├── support/
│   ├── test_helpers.ex
│   ├── numerical_helpers.ex
│   └── backend_helpers.ex
├── nx_penalties/
│   ├── penalties_test.exs
│   ├── divergences_test.exs
│   ├── constraints_test.exs
│   ├── pipeline_test.exs
│   └── integration/
│       └── axon_test.exs
└── property/
    ├── penalties_property_test.exs
    └── generators.ex

examples/
├── basic_usage.exs
├── pipeline_composition.exs
├── axon_training.exs
└── curriculum_learning.exs
```

---

## Implementation Order (TDD)

Follow this exact order. For each module:
1. **Write tests first** (from spec documents)
2. **Run tests** (see them fail)
3. **Implement** (make tests pass)
4. **Refactor** (improve code quality)
5. **Verify** (`mix test`, `mix credo --strict`, `mix compile --warnings-as-errors`)

### Phase 1: Foundation

#### Step 1: Test Support Modules
Create test helpers FIRST (copy from `07_TEST_STRATEGY.md`):
- `test/support/test_helpers.ex` - `assert_close/3`, `assert_scalar/1`, `assert_finite/1`
- `test/support/numerical_helpers.ex` - `verify_gradients/3`, `check_stability/3`
- `test/support/backend_helpers.ex` - `available_backends/0`, `with_backends/1`
- `test/test_helper.exs` - ExUnit configuration

#### Step 2: Penalties Module
Read: `01_PENALTY_PRIMITIVES.md`, ADR-001, ADR-002, ADR-003

1. Write `test/nx_penalties/penalties_test.exs` with all test cases from spec
2. Implement `lib/nx_penalties/penalties.ex`:
   - `defn l1/2` - L1 penalty (lambda default: 1.0)
   - `defn l2/2` - L2 penalty with optional clipping
   - `defn elastic_net/2` - Combined L1+L2
3. Implement `lib/nx_penalties/penalties/validation.ex` - NimbleOptions schemas
4. Write property tests in `test/property/penalties_property_test.exs`

**Key constraints:**
- All functions are `defn` (JIT-compatible)
- `lambda` defaults to `1.0` (unscaled primitives)
- Reduction options: `:sum`, `:mean`
- All functions return scalar tensors

#### Step 3: Divergences Module
Read: `02_DIVERGENCES.md`, ADR-004, ADR-005

1. Write `test/nx_penalties/divergences_test.exs`
2. Implement `lib/nx_penalties/divergences.ex`:
   - `defn kl_divergence/3` - KL(P||Q) with log-prob inputs
   - `defn js_divergence/3` - Jensen-Shannon (symmetric)
   - `defn entropy/2` - Shannon entropy with `:penalty`/`:bonus` mode
3. Implement validation module

**Key constraints:**
- Inputs are log-probabilities (not raw probabilities)
- Numerical stability with epsilon clamping
- Reduction options: `:mean`, `:sum`, `:none`

#### Step 4: Pipeline Module
Read: `04_PIPELINE.md`

1. Write `test/nx_penalties/pipeline_test.exs`
2. Implement `lib/nx_penalties/pipeline.ex`:
   - `%Pipeline{}` struct with entries as 5-tuple: `{name, fn, weight, opts, enabled}`
   - `new/1` - Create empty pipeline
   - `add/4` - Add penalty with weight
   - `remove/2` - Remove by name
   - `update_weight/3` - Modify weight
   - `set_enabled/3` - Enable/disable
   - `compute/3` - Execute and return `{total, metrics}`

**Key constraints:**
- Entries are 5-tuple: `{name, penalty_fn, weight, opts, enabled}`
- `compute/3` takes opts with `:extra_args` and `:track_grad_norms`
- Metrics map includes raw, weighted, and total values

#### Step 5: Main Entry Module
Read: `08_API_REFERENCE.md` "Hot Path vs Entry Point" section

1. Implement `lib/nx_penalties.ex`:
   - Validated wrappers: `l1/2`, `l2/2`, `elastic_net/2`, etc.
   - `pipeline/1` - Declarative pipeline builder
   - `compute/3` - Delegates to Pipeline.compute

**Key pattern:**
```elixir
def l1(tensor, opts \\ []) do
  opts = NxPenalties.Penalties.Validation.validate_l1!(opts)
  NxPenalties.Penalties.l1(tensor, opts)
end
```

#### Step 6: Telemetry
1. Implement `lib/nx_penalties/telemetry.ex`:
   - Events: `[:nx_penalties, :penalty, :compute, :start/:stop]`
   - Events: `[:nx_penalties, :pipeline, :compute, :start/:stop]`

#### Step 7: Axon Integration
Read: `05_AXON_INTEGRATION.md`

1. Write `test/nx_penalties/integration/axon_test.exs`
2. Implement `lib/nx_penalties/integration/axon.ex`:
   - `wrap_loss/3` - Add penalty to base loss
   - `wrap_loss_with_pipeline/3` - Pipeline-based loss wrapper
   - `capture_activation/2` - Activity regularization helper

### Phase 2: Stubs (Mark as v0.2)

#### Step 8: Constraints (Stubs)
Read: `03_CONSTRAINTS.md`

1. Implement `lib/nx_penalties/constraints.ex` with:
   - `defn orthogonality/2` - Basic implementation
   - `defn consistency/3` - Basic MSE-based
   - Document as "v0.2: Advanced options coming"

#### Step 9: Gradient Tracker (Stub)
Read: `11_GRADIENT_TRACKING.md`

1. Implement `lib/nx_penalties/gradient_tracker.ex`:
   - `compute_grad_norm/2` - Using `Nx.Defn.grad/1` correctly
   - `pipeline_grad_norms/2` - Per-penalty norms
   - `total_grad_norm/2` - Combined norm
   - Returns `nil` on non-differentiable operations

#### Step 10: Polaris Integration (Stub)
1. Implement `lib/nx_penalties/integration/polaris.ex`:
   - `add_l1_decay/2`, `add_l2_decay/2` - Gradient transforms
   - Document as "v0.2: Full implementation coming"

---

## Testing Requirements

### Supertester Patterns (Apply Throughout)

From `07_TEST_STRATEGY.md`:

1. **Zero Sleep** - No `Process.sleep`, use proper synchronization
2. **Parallel Execution** - All tests use `async: true`
3. **Isolation** - No shared state between tests
4. **Deterministic** - Seed random generators, avoid timing dependencies
5. **Expressive Assertions** - Use `assert_close/3`, `assert_finite/1`

### Required Test Coverage

| Module | Coverage Target |
|--------|-----------------|
| `NxPenalties.Penalties` | 100% |
| `NxPenalties.Divergences` | 100% |
| `NxPenalties.Pipeline` | 95% |
| `NxPenalties.Constraints` | 90% |
| `NxPenalties.Integration.*` | 90% |

### Test Categories for Each Function

1. **Correctness** - Known input → expected output
2. **Edge cases** - Zero, empty, negative, very large values
3. **Gradients** - `verify_gradients/3` for all differentiable functions
4. **JIT** - `assert_jit_compiles/2` for all `defn` functions
5. **Numerical stability** - `check_stability/3` across value ranges
6. **Properties** - Invariants (e.g., "L1 is always non-negative")

---

## Critical Implementation Details

### Lambda vs Weight Semantics

From `08_API_REFERENCE.md`:

| Concept | Default | Purpose |
|---------|---------|---------|
| `lambda` | `1.0` | Intrinsic penalty scaling (rarely changed) |
| `weight` | `1.0` | Pipeline combination weight (primary knob) |

**Correct usage:**
```elixir
# Good: weight is the scaling knob
pipeline = NxPenalties.pipeline([
  {:l1, weight: 0.001},
  {:l2, weight: 0.01}
])

# Avoid: double-scaling
{:l1, weight: 0.01, opts: [lambda: 0.1]}  # Confusing!
```

### Pipeline Entry 5-Tuple

From `04_PIPELINE.md`:

```elixir
@type entry :: {atom(), function(), number() | Nx.Tensor.t(), keyword(), boolean()}
#               name    penalty_fn   weight                      opts      enabled
```

### Nx.Defn.grad/1 Usage

From `11_GRADIENT_TRACKING.md`:

```elixir
# CORRECT: grad/1 returns a function
grad_fn = Nx.Defn.grad(loss_fn)
grad_tensor = grad_fn.(tensor)

# WRONG: This doesn't work
# grad_tensor = Nx.Defn.grad(tensor, loss_fn)  # Invalid!
```

### Numerical Stability Patterns

From `09_NUMERICAL_STABILITY.md`:

```elixir
# KL divergence - clamp before log
defn safe_kl(p_logprobs, q_logprobs) do
  p = Nx.exp(p_logprobs)
  diff = Nx.subtract(p_logprobs, q_logprobs)
  # Clamp to avoid -inf * 0
  clamped_p = Nx.max(p, 1.0e-10)
  Nx.sum(Nx.multiply(clamped_p, diff))
end

# Entropy - handle zero probabilities
defn safe_entropy(logprobs) do
  probs = Nx.exp(logprobs)
  # Where prob is tiny, contribution is ~0
  neg_plogp = Nx.multiply(Nx.negate(probs), logprobs)
  # Replace NaN (from 0 * -inf) with 0
  Nx.select(Nx.is_nan(neg_plogp), 0.0, neg_plogp)
  |> Nx.sum()
end
```

---

## Examples to Create

### 1. `examples/basic_usage.exs`
```elixir
# Demonstrate individual penalties
tensor = Nx.tensor([1.0, -2.0, 3.0, -0.5])

# L1 penalty
l1 = NxPenalties.l1(tensor)
IO.puts("L1: #{Nx.to_number(l1)}")

# L2 penalty
l2 = NxPenalties.l2(tensor)
IO.puts("L2: #{Nx.to_number(l2)}")

# Elastic Net
elastic = NxPenalties.elastic_net(tensor, l1_ratio: 0.7)
IO.puts("Elastic Net: #{Nx.to_number(elastic)}")
```

### 2. `examples/pipeline_composition.exs`
```elixir
# Build and execute a pipeline
pipeline = NxPenalties.pipeline([
  {:l1, weight: 0.001},
  {:l2, weight: 0.01},
  {:entropy, weight: 0.1, opts: [mode: :bonus]}
])

tensor = Nx.random_uniform({100, 50})
{total, metrics} = NxPenalties.compute(pipeline, tensor)

IO.puts("Total penalty: #{Nx.to_number(total)}")
IO.inspect(metrics, label: "Metrics")
```

### 3. `examples/axon_training.exs`
```elixir
# Integrate with Axon training loop
model = Axon.input("x", shape: {nil, 10})
|> Axon.dense(32, activation: :relu)
|> Axon.dense(1)

pipeline = NxPenalties.pipeline([{:l2, weight: 0.001}])

wrapped_loss = NxPenalties.Integration.Axon.wrap_loss_with_pipeline(
  &Axon.Losses.mean_squared_error/2,
  pipeline
)

# Training loop would use wrapped_loss
```

### 4. `examples/curriculum_learning.exs`
```elixir
# Dynamic weight adjustment during training
defmodule CurriculumTraining do
  def update_for_epoch(pipeline, epoch, max_epochs) do
    # Increase regularization over time
    progress = epoch / max_epochs

    pipeline
    |> NxPenalties.Pipeline.update_weight(:l2, 0.001 * (1 + progress))
    |> NxPenalties.Pipeline.update_weight(:kl, 0.1 * progress)
  end
end
```

---

## Final Verification Checklist

Before considering the implementation complete:

```bash
# All tests pass
mix test
# Expected: 0 failures

# No compiler warnings
mix compile --warnings-as-errors
# Expected: Compiles successfully

# Code quality
mix credo --strict
# Expected: 0 issues

# Formatting
mix format --check-formatted
# Expected: Already formatted

# Documentation builds
mix docs
# Expected: Generates docs successfully

# Examples run
mix run examples/basic_usage.exs
mix run examples/pipeline_composition.exs
# Expected: No errors, expected output
```

---

## Implementation Notes

### Do NOT:
- Add features not in the specs
- Create documentation files beyond what's needed
- Over-engineer or add "improvements"
- Use `Nx.pow` (use `Nx.power` for consistency)
- Return `0.0` for gradient failures (return `nil`)

### DO:
- Follow specs exactly as written
- Use the exact type signatures from `08_API_REFERENCE.md`
- Include all test cases from spec documents
- Add `@moduledoc`, `@doc`, and `@spec` to all public functions
- Emit telemetry events as specified
- Handle edge cases (empty tensors, NaN, Inf)

---

## Success Criteria

The implementation is complete when:

1. `mix test` - All tests pass (0 failures)
2. `mix compile --warnings-as-errors` - No warnings
3. `mix credo --strict` - No issues
4. `mix format` - Properly formatted
5. Examples in `examples/` run without errors
6. All public functions have `@doc`, `@spec`, and working examples
7. Coverage meets targets (use `mix coveralls.html` to verify)
