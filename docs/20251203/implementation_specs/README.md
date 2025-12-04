# NxPenalties Implementation Specifications

**Version**: 0.1.0 (Target)
**Date**: 2025-12-03
**Status**: Implementation Planning

## Overview

NxPenalties is a standalone Elixir library providing composable regularization penalties and loss functions for the Nx ecosystem. It fills a critical gap between Axon (model graph) and Polaris (optimization) by providing the "missing middleware" for complex training objectives.

## Quick Example

```elixir
# Simple penalty
l1_loss = NxPenalties.l1(params.weights)

# Pipeline composition (recommended)
pipeline = NxPenalties.pipeline([
  {:l1, weight: 0.001},    # Sparsity
  {:l2, weight: 0.01},     # Weight decay
  {:entropy, weight: 0.1, opts: [mode: :bonus]}  # Exploration
])

{total_penalty, metrics} = NxPenalties.compute(pipeline, model_outputs)

# Integrate with your training loop
total_loss = base_loss + total_penalty
```

## Document Index

| Document | Purpose | Priority |
|----------|---------|----------|
| [00_ARCHITECTURE.md](./00_ARCHITECTURE.md) | Core architecture, module structure, design decisions | Critical |
| [01_PENALTY_PRIMITIVES.md](./01_PENALTY_PRIMITIVES.md) | L1, L2, Elastic Net implementation specs | Phase 1 |
| [02_DIVERGENCES.md](./02_DIVERGENCES.md) | KL divergence, JS divergence, entropy | Phase 1 |
| [03_CONSTRAINTS.md](./03_CONSTRAINTS.md) | Orthogonality, gradient penalty, consistency | Phase 2 |
| [04_PIPELINE.md](./04_PIPELINE.md) | Composition, weighting, multi-objective | Phase 1 |
| [05_AXON_INTEGRATION.md](./05_AXON_INTEGRATION.md) | Axon.Loop helpers, layer wrappers | Phase 1 |
| [06_POLARIS_INTEGRATION.md](./06_POLARIS_INTEGRATION.md) | Gradient transform wrappers | Phase 2 |
| [07_TEST_STRATEGY.md](./07_TEST_STRATEGY.md) | TDD approach, supertester patterns | Critical |
| [08_API_REFERENCE.md](./08_API_REFERENCE.md) | Complete function signatures | Reference |
| [09_NUMERICAL_STABILITY.md](./09_NUMERICAL_STABILITY.md) | Stability patterns, edge cases | Critical |
| [10_BACKEND_COMPATIBILITY.md](./10_BACKEND_COMPATIBILITY.md) | EXLA/Torchx testing matrix | Critical |
| [11_GRADIENT_TRACKING.md](./11_GRADIENT_TRACKING.md) | Gradient norm monitoring (from tinkex) | Phase 2 |

## Strategic Context

### Why Standalone Library?

1. **Axon explicitly rejects model-level regularization** - Sean Moriarity removed regularization APIs, stating: "Regularization is a concern of training/optimization and not the model."

2. **Polaris is architecturally blind to activations** - Operates on `{params, gradients, state}` tuples; cannot see intermediate layer outputs needed for activity regularization.

3. **Scholar is traditional ML only** - Regularization is tightly coupled to specific estimators, not exposed as composable building blocks.

4. **Ecosystem precedent** - NxImage, NxSignal, Bumblebee all follow standalone pattern with independent release cycles.

### Core Constraints

| Constraint | Implication |
|------------|-------------|
| **All math in `Nx.Defn`** | Enables JIT compilation, GPU execution |
| **Backend agnostic** | Must work on EXLA, Torchx, BinaryBackend |
| **Immutable state** | Cannot use side-effects; must thread auxiliary values |
| **Composable** | Functions combine without special glue code |

## Implementation Phases

### Phase 1: MVP (v0.1)
- Core penalties: L1, L2, Elastic Net
- Divergences: KL, entropy
- Pipeline composition with weights
- Basic Axon.Loop integration
- Comprehensive test suite
- Hex.pm publication

### Phase 2: Extended (v0.2)
- Constraints: orthogonality, consistency
- Polaris gradient transforms
- Activity regularization via layer wrappers
- Advanced telemetry
- Livebook examples
- Gradient penalty (advanced, expensive)
- Gradient tracking / norm monitoring (N+1 backward passes)

### Phase 3: Advanced (v0.3+)
- Auxiliary loss infrastructure
- Multi-head output support
- Pipeline.Multi for named inputs
- Containerized loss pattern
- Community feedback integration

## Dependencies

```elixir
defp deps do
  [
    # Core
    {:nx, "~> 0.9"},
    {:nimble_options, "~> 1.0"},
    {:telemetry, "~> 1.0"},

    # Test only
    {:exla, ">= 0.0.0", only: :test},
    {:supertester, "~> 0.3", only: :test},

    # Docs only
    {:ex_doc, "~> 0.34", only: :docs}
  ]
end
```

## Module Structure

```
lib/
└── nx_penalties/
    ├── penalties.ex           # L1, L2, elastic_net
    ├── divergences.ex         # kl, js, entropy
    ├── constraints.ex         # orthogonality, gradient_penalty
    ├── pipeline.ex            # Composition engine
    ├── gradient_tracker.ex    # Gradient norm monitoring
    ├── telemetry.ex           # Instrumentation
    └── integration/
        ├── axon.ex            # Axon.Loop helpers
        └── polaris.ex         # Gradient transforms
```

## Success Criteria

1. **Functional**: All regularizers produce correct numerical output
2. **Performant**: JIT-compiled, GPU-accelerated when available
3. **Stable**: Numerically robust across edge cases
4. **Tested**: 100% coverage with property-based tests
5. **Documented**: Livebook examples, hexdocs, type specs
6. **Adopted**: Published to Hex.pm, used by downstream projects
